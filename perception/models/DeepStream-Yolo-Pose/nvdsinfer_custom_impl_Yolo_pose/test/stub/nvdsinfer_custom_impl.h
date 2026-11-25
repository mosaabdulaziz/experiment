// A TEST STUB of DeepStream's nvdsinfer_custom_impl.h.
//
// The parser is compiled into a shared library inside the DeepStream container, where the real
// header lives. This file lets the SAME parser source be compiled and run on a plain host, with
// no DeepStream and no GPU -- which is how the parser is proved (FALL_DETECTION_DESIGN.md §11:
// the parser is fed a tensor whose contents are known).
//
// The declarations below are the ones the parser uses, plus the structs that hold them, copied
// from DeepStream 9.0's nvdsinfer.h and nvdsinfer_custom_impl.h under their REAL names, so that a
// parser written against this stub compiles against the real header too. If the API drifts, this
// stub is where the drift shows.
//
// The real names matter: DeepStream 8.0 renamed the dims constant to NVDSINFER_MAX_DIMS and the
// class count to numClassesConfigured, and the stub carries the renamed ones, not the old.
//
// Deliberately left out: NvDsInferParseDetectionParams' deprecated perClassThreshold member,
// which the parser does not read.
//
// The real header is NOT used, and nothing here ships: the image builds the parser against
// /opt/nvidia/deepstream/deepstream/custom_parsers/nvdsinfer_custom_impl.h.

#ifndef NVDSINFER_CUSTOM_IMPL_STUB_H
#define NVDSINFER_CUSTOM_IMPL_STUB_H

#include <vector>

#define NVDSINFER_MAX_DIMS 8

// glib's typedef, which the real header's include chain provides (nvdsinfer.h pulls in glib).
typedef unsigned int uint;

typedef enum {
  FLOAT = 0,
  HALF = 1,
  INT8 = 2,
  INT32 = 3,
  BOOL = 4,
  FP16 = 5,
  BF16 = 6
} NvDsInferDataType;

typedef struct {
  unsigned int numDims;
  unsigned int d[NVDSINFER_MAX_DIMS];
  unsigned int numElements;
} NvDsInferDims;

typedef struct {
  NvDsInferDataType dataType;
  NvDsInferDims inferDims;
  int bindingIndex;
  const char* layerName;
  void* buffer;
  int isInput;
} NvDsInferLayerInfo;

typedef struct {
  unsigned int width;
  unsigned int height;
  unsigned int channels;
} NvDsInferNetworkInfo;

typedef struct {
  unsigned int numClassesConfigured;
  std::vector<float> perClassPreclusterThreshold;
  std::vector<float> perClassPostclusterThreshold;
} NvDsInferParseDetectionParams;

typedef struct {
  unsigned int classId;
  float left;
  float top;
  float width;
  float height;
  float detectionConfidence;
  float* mask;
  unsigned int mask_width;
  unsigned int mask_height;
  unsigned int mask_size;
} NvDsInferInstanceMaskInfo;

// The real macro also emits a compile-time check of the function's signature. The declaration is
// the part the parser depends on, and it is what this stub reproduces.
#define CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(customParseFunc) \
  extern "C" bool customParseFunc(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, \
      NvDsInferNetworkInfo const& networkInfo, \
      NvDsInferParseDetectionParams const& detectionParams, \
      std::vector<NvDsInferInstanceMaskInfo>& objectList);

#endif  // NVDSINFER_CUSTOM_IMPL_STUB_H
