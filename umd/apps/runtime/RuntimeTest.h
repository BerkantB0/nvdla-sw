/*
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#include "dlaerror.h"
#include "dlatypes.h"

#include "nvdla/IRuntime.h"

#include "DlaImage.h"
#include "ErrorMacros.h"

#include "nvdla_inf.h"

#include <string>
#include <vector>

struct PerformanceSample
{
    NvU32 index;
    bool warmup;
    NvU64 runtimeExecutionNs;
    NvU64 outputExtractNs;
};

struct PerformanceProfile
{
    PerformanceProfile() :
        clockResolutionNs(0),
        clockPairOverheadNs(0),
        runtimeCreateNs(0),
        loadableReadNs(0),
        runtimeLoadNs(0),
        emuInitNs(0),
        inputSetupNs(0),
        outputSetupNs(0),
        outputWriteNs(0),
        bufferCleanupNs(0),
        emuStopNs(0),
        runtimeUnloadNs(0),
        runtimeDestroyNs(0),
        testTotalNs(0),
        processTotalNs(0),
        outputsConsistent(true),
        status(NvDlaSuccess)
    {}

    NvU64 clockResolutionNs;
    NvU64 clockPairOverheadNs;
    NvU64 runtimeCreateNs;
    NvU64 loadableReadNs;
    NvU64 runtimeLoadNs;
    NvU64 emuInitNs;
    NvU64 inputSetupNs;
    NvU64 outputSetupNs;
    NvU64 outputWriteNs;
    NvU64 bufferCleanupNs;
    NvU64 emuStopNs;
    NvU64 runtimeUnloadNs;
    NvU64 runtimeDestroyNs;
    NvU64 testTotalNs;
    NvU64 processTotalNs;
    bool outputsConsistent;
    NvDlaError status;
    std::vector<PerformanceSample> samples;
};

struct TestAppArgs
{
    std::string inputPath;
    std::string inputName;
    std::string loadableName;
    NvS32 serverPort;
    NvU8 normalize_value;
    float mean[4];
    bool rawOutputDump;
    std::string profilePath;
    NvU32 warmupIterations;
    NvU32 measuredIterations;

    TestAppArgs() :
        inputPath("./"),
        inputName(""),
        loadableName(""),
        serverPort(6666),
        normalize_value(1),
        mean{0.0, 0.0, 0.0, 0.0},
        rawOutputDump(false),
        profilePath(""),
        warmupIterations(0),
        measuredIterations(1)
    {}
};

struct TestInfo
{
    TestInfo() :
        runtime(NULL),
        inputLoadablePath(""),
        inputHandle(NULL),
        outputHandle(NULL),
        pData(NULL),
        dlaServerRunning(false),
        dlaRemoteSock(-1),
        dlaServerSock(-1),
        numInputs(0),
        numOutputs(0),
        inputImage(NULL),
        outputImage(NULL),
        profile()
    {}
    // runtime
    nvdla::IRuntime* runtime;
    std::string inputLoadablePath;
    NvU8 *inputHandle;
    NvU8 *outputHandle;
    NvU8 *pData;
    bool dlaServerRunning;
    NvS32 dlaRemoteSock;
    NvS32 dlaServerSock;
    NvU32 numInputs;
    NvU32 numOutputs;
    NvDlaImage* inputImage;
    NvDlaImage* outputImage;
    PerformanceProfile profile;
};

// Test
NvDlaError run(const TestAppArgs* appArgs, TestInfo* i);
