/*
 * Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "DlaImageUtils.h"
#include "ErrorMacros.h"
#include "RuntimeTest.h"

#include <time.h>
#include <sys/time.h>
#include <stdio.h>

#include "nvdla/IRuntime.h"

#include "half.h"
#include "main.h"
#include "nvdla_os_inf.h"

#include "dlaerror.h"
#include "dlatypes.h"

#include <cstdio> // snprintf, fopen
#include <cstring>
#include <string>
#include <vector>

#define OUTPUT_DIMG "output.dimg"

using namespace half_float;

static NvU64 monotonicRawNs()
{
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC_RAW, &value);
    return (NvU64)value.tv_sec * 1000000000ULL + (NvU64)value.tv_nsec;
}

static NvU64 elapsedNs(NvU64 before)
{
    return monotonicRawNs() - before;
}

static NvU64 measureClockPairOverheadNs()
{
    NvU64 minimum = ~0ULL;

    for (NvU32 index = 0; index < 1000; ++index)
    {
        NvU64 before = monotonicRawNs();
        NvU64 overhead = elapsedNs(before);
        if (overhead < minimum)
            minimum = overhead;
    }
    return minimum;
}

static bool writePerformanceProfile(const TestAppArgs* appArgs,
                                    const TestInfo* i)
{
    FILE* output;
    const PerformanceProfile& profile = i->profile;

    if (appArgs->profilePath.empty())
        return true;

    output = fopen(appArgs->profilePath.c_str(), "w");
    if (!output)
        return false;

    fprintf(output,
            "{\n"
            "  \"schema_version\": 2,\n"
            "  \"clock\": \"CLOCK_MONOTONIC_RAW\",\n"
            "  \"clock_resolution_ns\": %llu,\n"
            "  \"clock_pair_overhead_ns\": %llu,\n"
            "  \"warmup_iterations\": %u,\n"
            "  \"measured_iterations\": %u,\n"
            "  \"outputs_consistent\": %s,\n"
            "  \"status\": %d,\n"
            "  \"phases_ns\": {\n"
            "    \"runtime_create\": %llu,\n"
            "    \"loadable_read\": %llu,\n"
            "    \"runtime_load\": %llu,\n"
            "    \"emu_init\": %llu,\n"
            "    \"input_setup\": %llu,\n"
            "    \"output_setup\": %llu,\n"
            "    \"output_write\": %llu,\n"
            "    \"buffer_cleanup\": %llu,\n"
            "    \"emu_stop\": %llu,\n"
            "    \"runtime_unload\": %llu,\n"
            "    \"runtime_destroy\": %llu,\n"
            "    \"test_total\": %llu,\n"
            "    \"process_total\": %llu\n"
            "  },\n"
            "  \"samples\": [\n",
            (unsigned long long)profile.clockResolutionNs,
            (unsigned long long)profile.clockPairOverheadNs,
            appArgs->warmupIterations,
            appArgs->measuredIterations,
            profile.outputsConsistent ? "true" : "false",
            profile.status,
            (unsigned long long)profile.runtimeCreateNs,
            (unsigned long long)profile.loadableReadNs,
            (unsigned long long)profile.runtimeLoadNs,
            (unsigned long long)profile.emuInitNs,
            (unsigned long long)profile.inputSetupNs,
            (unsigned long long)profile.outputSetupNs,
            (unsigned long long)profile.outputWriteNs,
            (unsigned long long)profile.bufferCleanupNs,
            (unsigned long long)profile.emuStopNs,
            (unsigned long long)profile.runtimeUnloadNs,
            (unsigned long long)profile.runtimeDestroyNs,
            (unsigned long long)profile.testTotalNs,
            (unsigned long long)profile.processTotalNs);

    for (size_t index = 0; index < profile.samples.size(); ++index)
    {
        const PerformanceSample& sample = profile.samples[index];
        fprintf(output,
                "    {\"index\": %u, \"warmup\": %s, "
                "\"runtime_execution_ns\": %llu, \"output_extract_ns\": %llu}%s\n",
                sample.index,
                sample.warmup ? "true" : "false",
                (unsigned long long)sample.runtimeExecutionNs,
                (unsigned long long)sample.outputExtractNs,
                index + 1 == profile.samples.size() ? "" : ",");
    }
    fprintf(output, "  ]\n}\n");

    return fclose(output) == 0;
}

static TestImageTypes getImageType(std::string imageFileName)
{
    TestImageTypes it = IMAGE_TYPE_UNKNOWN;
    std::string ext = imageFileName.substr(imageFileName.find_last_of(".") + 1);
    if (ext == "pgm")
    {
        it = IMAGE_TYPE_PGM;
    }
    else if (ext == "jpg")
    {
        it = IMAGE_TYPE_JPG;
    }

    return it;
}

static NvDlaError copyImageToInputTensor
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pImgBuffer,
    nvdla::IRuntime::NvDlaTensor *tensorDesc
)
{
    NvDlaError e = NvDlaSuccess;

    std::string imgPath = /*i->inputImagesPath + */appArgs->inputName;
    NvDlaImage* R8Image = new NvDlaImage();
    NvDlaImage* tensorImage = NULL;
    TestImageTypes imageType = getImageType(imgPath);
    if (!R8Image)
        ORIGINATE_ERROR(NvDlaError_InsufficientMemory);

    switch(imageType) {
        case IMAGE_TYPE_PGM:
            PROPAGATE_ERROR(PGM2DIMG(imgPath, R8Image, tensorDesc));
            break;
        case IMAGE_TYPE_JPG:
            PROPAGATE_ERROR(JPEG2DIMG(imgPath, R8Image, tensorDesc));
            break;
        default:
            //TODO Fix this error condition
//          ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Unknown image type: %s", imgPath.c_str());
            NvDlaDebugPrintf("Unknown image type: %s", imgPath.c_str());
            goto fail;
    }

    tensorImage = i->inputImage;
    if (tensorImage == NULL)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "NULL input Image");

    PROPAGATE_ERROR(createImageCopy(appArgs, R8Image, tensorDesc, tensorImage));

    //tensorImage->printBuffer(true);  /* Print the input Buffer */ 

    PROPAGATE_ERROR(DIMG2DlaBuffer(tensorImage, pImgBuffer));

fail:
    if (R8Image != NULL && R8Image->m_pData != NULL)
        NvDlaFree(R8Image->m_pData);
    delete R8Image;

    return e;
}

static NvDlaError prepareOutputTensor
(
    nvdla::IRuntime::NvDlaTensor* pTDesc,
    NvDlaImage* pOutImage,
    void** pOutBuffer,
    const TestAppArgs* appArgs
)
{
    NvDlaError e = NvDlaSuccess;

    PROPAGATE_ERROR_FAIL(Tensor2DIMG(appArgs, pTDesc, pOutImage));
    PROPAGATE_ERROR_FAIL(DIMG2DlaBuffer(pOutImage, pOutBuffer));

fail:
    return e;
}


NvDlaError setupInputBuffer
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pInputBuffer
)
{
    NvDlaError e = NvDlaSuccess;
    void *hMem = NULL;
    NvS32 numInputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    PROPAGATE_ERROR_FAIL(runtime->getNumInputTensors(&numInputTensors));

    i->numInputs = numInputTensors;

    if (numInputTensors < 1)
        goto fail;

    PROPAGATE_ERROR_FAIL(runtime->getInputTensorDesc(0, &tDesc));

    PROPAGATE_ERROR_FAIL(runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, pInputBuffer));
    i->inputHandle = (NvU8 *)hMem;
    PROPAGATE_ERROR_FAIL(copyImageToInputTensor(appArgs, i, pInputBuffer, &tDesc));

    if (!runtime->bindInputTensor(0, hMem))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindInputTensor() failed");

fail:
    return e;
}

static void cleanupInputBuffer(const TestAppArgs *appArgs,
                                TestInfo *i)
{
    nvdla::IRuntime *runtime = NULL;
    NvS32 numInputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    NvDlaError e = NvDlaSuccess;

    if (i->inputImage != NULL && i->inputImage->m_pData != NULL) {
        NvDlaFree(i->inputImage->m_pData);
        i->inputImage->m_pData = NULL;
    }

    runtime = i->runtime;
    if (runtime == NULL)
        return;
    e = runtime->getNumInputTensors(&numInputTensors);
    if (e != NvDlaSuccess)
        return;

    if (numInputTensors < 1)
        return;

    e = runtime->getInputTensorDesc(0, &tDesc);
    if (e != NvDlaSuccess)
        return;

    if (i->inputHandle == NULL)
        return;

    /* Free the buffer allocated */
    runtime->freeSystemMemory(i->inputHandle, tDesc.bufferSize);
    i->inputHandle = NULL;
    return;
}

NvDlaError setupOutputBuffer
(
    const TestAppArgs* appArgs,
    TestInfo* i,
    void** pOutputBuffer
)
{
    NVDLA_UNUSED(appArgs);

    NvDlaError e = NvDlaSuccess;
    void *hMem;
    NvS32 numOutputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    NvDlaImage *pOutputImage = NULL;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    PROPAGATE_ERROR_FAIL(runtime->getNumOutputTensors(&numOutputTensors));

    i->numOutputs = numOutputTensors;

    if (numOutputTensors < 1)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "Expected number of output tensors of %u, found %u", 1, numOutputTensors);

    PROPAGATE_ERROR_FAIL(runtime->getOutputTensorDesc(0, &tDesc));
    PROPAGATE_ERROR_FAIL(runtime->allocateSystemMemory(&hMem, tDesc.bufferSize, pOutputBuffer));
    i->outputHandle = (NvU8 *)hMem;

    pOutputImage = i->outputImage;
    if (i->outputImage == NULL)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "NULL Output image");
    PROPAGATE_ERROR_FAIL(prepareOutputTensor(&tDesc, pOutputImage, pOutputBuffer, appArgs));

    if (!runtime->bindOutputTensor(0, hMem))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->bindOutputTensor() failed");

fail:
    return e;
}

static void cleanupOutputBuffer(const TestAppArgs *appArgs,
                                TestInfo *i)
{
    nvdla::IRuntime *runtime = NULL;
    NvS32 numOutputTensors = 0;
    nvdla::IRuntime::NvDlaTensor tDesc;
    NvDlaError e = NvDlaSuccess;

    /* Do not clear outputImage if in server mode */
    if (!i->dlaServerRunning &&
            i->outputImage != NULL &&
            i->outputImage->m_pData != NULL) {
        NvDlaFree(i->outputImage->m_pData);
        i->outputImage->m_pData = NULL;
    }

    runtime = i->runtime;
    if (runtime == NULL)
        return;
    e = runtime->getNumOutputTensors(&numOutputTensors);
    if (e != NvDlaSuccess)
        return;
    e = runtime->getOutputTensorDesc(0, &tDesc);
    if (e != NvDlaSuccess)
        return;

    if (i->outputHandle == NULL)
        return;

    /* Free the buffer allocated */
    runtime->freeSystemMemory(i->outputHandle, tDesc.bufferSize);
    i->outputHandle = NULL;
    return;
}

static NvDlaError readLoadable(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    NVDLA_UNUSED(appArgs);
    std::string loadableName;
    NvDlaFileHandle file;
    NvDlaStatType finfo;
    size_t file_size;
    NvU8 *buf = 0;
    size_t actually_read = 0;
    NvDlaError rc;

    // Determine loadable path
    if (appArgs->loadableName == "")
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_NotInitialized, "No loadable found to load");
    }

    loadableName = appArgs->loadableName;

    rc = NvDlaFopen(loadableName.c_str(), NVDLA_OPEN_READ, &file);
    if (rc != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "couldn't open %s\n", loadableName.c_str());
    }

    rc = NvDlaFstat(file, &finfo);
    if ( rc != NvDlaSuccess)
    {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "couldn't get file stats for %s\n", loadableName.c_str());
    }

    file_size = NvDlaStatGetSize(&finfo);
    if ( !file_size ) {
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "zero-length for %s\n", loadableName.c_str());
    }

    buf = new NvU8[file_size];

    NvDlaFseek(file, 0, NvDlaSeek_Set);

    rc = NvDlaFread(file, buf, file_size, &actually_read);
    if ( rc != NvDlaSuccess )
    {
        NvDlaFree(buf);
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "read error for %s\n", loadableName.c_str());
    }
    NvDlaFclose(file);
    if ( actually_read != file_size ) {
        NvDlaFree(buf);
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "read wrong size for buffer? %d\n", actually_read);
    }

    i->pData = buf;

fail:
    return e;
}

NvDlaError loadLoadable(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    if (!runtime->load(i->pData, 0))
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "runtime->load failed");

fail:
    return e;
}

void unloadLoadable(const TestAppArgs* appArgs, TestInfo *i)
{
    NVDLA_UNUSED(appArgs);
    nvdla::IRuntime *runtime = NULL;

    runtime = i->runtime;
    if (runtime != NULL) {
        runtime->unload();
    }
}

double get_elapsed_time_seconds(struct timespec *before, struct timespec *after)
{
  double deltat_s  = after->tv_sec - before->tv_sec;
  double deltat_ns = (after->tv_nsec - before->tv_nsec) / 1000000000.0;
  return deltat_s + deltat_ns;
}

NvDlaError runTest(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    void* pInputBuffer = NULL;
    void* pOutputBuffer = NULL;
    NvU32 totalIterations =
        appArgs->warmupIterations + appArgs->measuredIterations;
    std::vector<NvU8> referenceOutput;
    NvU64 phaseStart;
    NvU64 testStart;

    nvdla::IRuntime* runtime = i->runtime;
    if (!runtime)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "getRuntime() failed");

    i->inputImage = new NvDlaImage();
    i->outputImage = new NvDlaImage();

    testStart = monotonicRawNs();
    phaseStart = monotonicRawNs();
    PROPAGATE_ERROR_FAIL(setupInputBuffer(appArgs, i, &pInputBuffer));
    i->profile.inputSetupNs = elapsedNs(phaseStart);

    phaseStart = monotonicRawNs();
    PROPAGATE_ERROR_FAIL(setupOutputBuffer(appArgs, i, &pOutputBuffer));
    i->profile.outputSetupNs = elapsedNs(phaseStart);
    NvDlaDebugPrintf("submitting tasks...\n");
    for (NvU32 index = 0; index < totalIterations; ++index)
    {
        PerformanceSample sample;
        sample.index = index + 1;
        sample.warmup = index < appArgs->warmupIterations;

        phaseStart = monotonicRawNs();
        if (!runtime->submit())
            ORIGINATE_ERROR(NvDlaError_BadParameter,
                            "runtime->submit() failed");
        sample.runtimeExecutionNs = elapsedNs(phaseStart);

        phaseStart = monotonicRawNs();
        PROPAGATE_ERROR_FAIL(DlaBuffer2DIMG(&pOutputBuffer,
                                            i->outputImage));
        sample.outputExtractNs = elapsedNs(phaseStart);

        const NvU8* outputBytes =
            static_cast<const NvU8*>(i->outputImage->m_pData);
        if (referenceOutput.empty())
        {
            referenceOutput.assign(outputBytes,
                                   outputBytes +
                                   i->outputImage->m_meta.size);
        }
        else if (referenceOutput.size() != i->outputImage->m_meta.size ||
                 memcmp(&referenceOutput[0],
                        outputBytes,
                        referenceOutput.size()) != 0)
        {
            i->profile.outputsConsistent = false;
        }

        i->profile.samples.push_back(sample);
        if (appArgs->profilePath.empty())
            NvDlaDebugPrintf("execution time = %f s\n",
                             sample.runtimeExecutionNs / 1000000000.0);
    }

    //i->outputImage->printBuffer(true);   /* Print the output buffer */

    /* Dump output dimg to a file */
    phaseStart = monotonicRawNs();
    PROPAGATE_ERROR_FAIL(DIMG2DIMGFile(i->outputImage,
                                        OUTPUT_DIMG,
                                        true,
                                        appArgs->rawOutputDump));
    i->profile.outputWriteNs = elapsedNs(phaseStart);
    i->profile.testTotalNs = elapsedNs(testStart);

fail:
    phaseStart = monotonicRawNs();
    cleanupOutputBuffer(appArgs, i);
    /* Do not clear outputImage if in server mode */
    if (!i->dlaServerRunning && i->outputImage != NULL) {
        delete i->outputImage;
        i->outputImage = NULL;
    }

    cleanupInputBuffer(appArgs, i);
    if (i->inputImage != NULL) {
        delete i->inputImage;
        i->inputImage = NULL;
    }
    i->profile.bufferCleanupNs = elapsedNs(phaseStart);

    return e;
}

NvDlaError run(const TestAppArgs* appArgs, TestInfo* i)
{
    NvDlaError e = NvDlaSuccess;
    NvU64 phaseStart;
    NvU64 processStart = monotonicRawNs();
    struct timespec resolution;

    if (clock_getres(CLOCK_MONOTONIC_RAW, &resolution) == 0)
        i->profile.clockResolutionNs =
            (NvU64)resolution.tv_sec * 1000000000ULL +
            (NvU64)resolution.tv_nsec;
    i->profile.clockPairOverheadNs = measureClockPairOverheadNs();

    /* Create runtime instance */
    NvDlaDebugPrintf("creating new runtime context...\n");
    phaseStart = monotonicRawNs();
    i->runtime = nvdla::createRuntime();
    i->profile.runtimeCreateNs = elapsedNs(phaseStart);
    if (i->runtime == NULL)
        ORIGINATE_ERROR_FAIL(NvDlaError_BadParameter, "createRuntime() failed");

    if (!i->dlaServerRunning)
    {
        phaseStart = monotonicRawNs();
        PROPAGATE_ERROR_FAIL(readLoadable(appArgs, i));
        i->profile.loadableReadNs = elapsedNs(phaseStart);
    }

    /* Load loadable */
    phaseStart = monotonicRawNs();
    PROPAGATE_ERROR_FAIL(loadLoadable(appArgs, i));
    i->profile.runtimeLoadNs = elapsedNs(phaseStart);

    /* Start emulator */
    phaseStart = monotonicRawNs();
    if (!i->runtime->initEMU())
        ORIGINATE_ERROR(NvDlaError_DeviceNotFound, "runtime->initEMU() failed");
    i->profile.emuInitNs = elapsedNs(phaseStart);

    /* Run test */
    PROPAGATE_ERROR_FAIL(runTest(appArgs, i));

fail:
    /* Stop emulator */
    phaseStart = monotonicRawNs();
    if (i->runtime != NULL)
        i->runtime->stopEMU();
    i->profile.emuStopNs = elapsedNs(phaseStart);

    /* Unload loadables */
    phaseStart = monotonicRawNs();
    unloadLoadable(appArgs, i);
    i->profile.runtimeUnloadNs = elapsedNs(phaseStart);

    /* Free if allocated in read Loadable */
    if (!i->dlaServerRunning && i->pData != NULL) {
        delete[] i->pData;
        i->pData = NULL;
    }

    /* Destroy runtime */
    phaseStart = monotonicRawNs();
    nvdla::destroyRuntime(i->runtime);
    i->profile.runtimeDestroyNs = elapsedNs(phaseStart);
    i->profile.status = e;
    i->profile.processTotalNs = elapsedNs(processStart);

    if (!writePerformanceProfile(appArgs, i) && e == NvDlaSuccess)
        e = NvDlaError_FileOperationFailed;

    return e;
}
