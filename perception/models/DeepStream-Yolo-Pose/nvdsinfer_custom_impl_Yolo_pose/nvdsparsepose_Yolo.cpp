// The pose parser: the exported graph's three tensors become one entry per person.
//
// The export (products/fall_detection/export_pose.py) writes three tensors, each self-describing:
//
//   boxes      [batch, N, 4]     x1, y1, x2, y2 in the network's input pixels
//   scores     [batch, N]        the detection confidence
//   keypoints  [batch, N, 17, 3] x, y, conf, in the network's input pixels
//
// The threshold and the NMS live here, not in the graph: the graph answers "what did the network
// see", the parser answers "what is a detection" (the face parser's arrangement).
//
// The payload lands in the object's instance-mask field -- 51 floats (17 joints x x, y, conf) plus
// the network's input size -- exactly as the face parser packs its five landmarks there. The pose
// module reads them back (perception/modules/pose/__init__.py) and turns the network's pixels into
// the frame's.
//
// HOW THE OUTPUTS ARRIVE. The export writes an explicit batch axis, because nvinfer batches the
// people's crops into one inference (batch-size in the config file). Whether the layer nvinfer
// hands the parser carries that axis is nvinfer's business, and this parser accepts BOTH:
//
//   per crop   boxes [N,4]   scores [N]   keypoints [N,17,3]
//   batched    boxes [B,N,4] scores [B,N] keypoints [B,N,17,3]
//
// The boxes layer decides which: its LAST axis is the 4 of a box in either case, so a two-axis
// layer is one crop's output and a three-axis layer is a batch of them. The other two must then
// have the shape that view implies. The batched view is read crop by crop -- each crop is a
// DIFFERENT person, in its OWN network-input pixels, so the NMS runs per crop (an NMS across
// crops would suppress one person with another).

#include <cassert>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#include "nvdsinfer_custom_impl.h"

#define NMS_THRESH 0.45

// COCO's 17 joints, each with x, y and its own confidence.
static const int KEYPOINT_COUNT = 17;
static const int KEYPOINT_DIMS = 3;
static const int KEYPOINT_VALUES = KEYPOINT_COUNT * KEYPOINT_DIMS;

// The export names the three tensors, and TensorRT keeps the ONNX output names, so the name is
// the first way to find a layer. The shape is the fallback: a builder may rewrite a name, but the
// three shapes cannot be mistaken for each other.
static const char* const BOXES_LAYER = "boxes";
static const char* const SCORES_LAYER = "scores";
static const char* const KEYPOINTS_LAYER = "keypoints";

// How the outputs arrived: one crop's output, or a batch of them.
struct Layout {
  bool batched;        // the layer carries the batch axis
  size_t count;        // detections in ONE crop
  size_t batch;        // crops in this inference
};

// The bounds on a real inference. The config builds the engine at a batch of 16 and the graph has
// 8400 candidates, so these are far above anything real: they exist to catch a dim nvinfer could
// not resolve, which comes as -1 and is 4294967295 as an unsigned count.
static const unsigned int MAX_BATCH = 4096;
static const unsigned int MAX_DETECTIONS = 1000000;

// The layout a layer's shape says, if it is a boxes layer at all.
static bool
boxesLayout(const NvDsInferLayerInfo& layer, Layout& layout)
{
  const NvDsInferDims& d = layer.inferDims;
  if (d.numDims == 2 && d.d[1] == 4) {
    layout = Layout{false, d.d[0], 1};
    return true;
  }
  if (d.numDims == 3 && d.d[2] == 4) {
    layout = Layout{true, d.d[1], d.d[0]};
    return true;
  }
  return false;
}

// The boxes layer: the one whose LAST axis is the 4 of a box.
//
// Two axes means one crop's output ([N,4]) and three axes mean a batch of them ([B,N,4]) -- the
// same layers with the batch axis stripped, which is the only difference between the two views.
// The NAME is the first way to find it, the way the other two are found; the shape is the
// fallback, for a builder that rewrites the name.
static const NvDsInferLayerInfo*
findBoxesLayer(const std::vector<NvDsInferLayerInfo>& layers, Layout& layout)
{
  for (const auto& layer : layers) {
    if (layer.layerName != nullptr && std::strcmp(layer.layerName, BOXES_LAYER) == 0
        && boxesLayout(layer, layout)) {
      return &layer;
    }
  }
  for (const auto& layer : layers) {
    if (boxesLayout(layer, layout)) {
      return &layer;
    }
  }
  return nullptr;
}

// The batch and the count must be real numbers. A dim nvinfer could not resolve comes as -1, which
// as an unsigned count is 4294967295 -- and the crop loop would then walk four billion crops off
// the end of the buffer.
static bool
layoutIsSane(const Layout& layout)
{
  return layout.count > 0 && layout.count <= MAX_DETECTIONS
      && layout.batch > 0 && layout.batch <= MAX_BATCH;
}

// The scores layer, as the layout says it must be: [N] for one crop, [B,N] for a batch.
static bool
isScoresShape(const NvDsInferLayerInfo& layer, const Layout& layout)
{
  const NvDsInferDims& d = layer.inferDims;
  if (!layout.batched) {
    // one crop's scores: [N], or the [N,1] the same data can be written as
    return (d.numDims == 1 && d.d[0] == layout.count)
        || (d.numDims == 2 && d.d[1] == 1 && d.d[0] == layout.count);
  }
  return d.numDims == 2 && d.d[0] == layout.batch && d.d[1] == layout.count;
}

// The keypoints layer, as the layout says it must be: [N,17,3] for one crop, [B,N,17,3] for a
// batch, or the same data flattened to [N,51] / [B,N,51].
static bool
isKeypointsShape(const NvDsInferLayerInfo& layer, const Layout& layout)
{
  const NvDsInferDims& d = layer.inferDims;
  const size_t joints = KEYPOINT_COUNT * KEYPOINT_DIMS;
  if (!layout.batched) {
    return (d.numDims == 3 && d.d[0] == layout.count
            && d.d[1] == KEYPOINT_COUNT && d.d[2] == KEYPOINT_DIMS)
        || (d.numDims == 2 && d.d[0] == layout.count && d.d[1] == joints);
  }
  return (d.numDims == 4 && d.d[0] == layout.batch && d.d[1] == layout.count
          && d.d[2] == KEYPOINT_COUNT && d.d[3] == KEYPOINT_DIMS)
      || (d.numDims == 3 && d.d[0] == layout.batch && d.d[1] == layout.count && d.d[2] == joints);
}

// The layer with this name whose shape fits, or -- when no layer carries the name -- the first
// layer whose shape fits.
template <typename Matches>
static const NvDsInferLayerInfo*
findLayer(const std::vector<NvDsInferLayerInfo>& layers, const char* name, Matches matches)
{
  for (const auto& layer : layers) {
    if (layer.layerName != nullptr && std::strcmp(layer.layerName, name) == 0 && matches(layer)) {
      return &layer;
    }
  }
  for (const auto& layer : layers) {
    if (matches(layer)) {
      return &layer;
    }
  }
  return nullptr;
}

extern "C" bool
NvDsInferParseYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList);

static float
clamp(float val, float minVal, float maxVal)
{
  assert(minVal <= maxVal);
  return std::min(maxVal, std::max(minVal, val));
}

static float
overlap1D(float x1min, float x1max, float x2min, float x2max)
{
  if (x1min > x2min) {
    std::swap(x1min, x2min);
    std::swap(x1max, x2max);
  }
  return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
}

static float
computeIoU(NvDsInferInstanceMaskInfo& bbox1, NvDsInferInstanceMaskInfo& bbox2)
{
  float overlapX = overlap1D(bbox1.left, bbox1.left + bbox1.width, bbox2.left, bbox2.left + bbox2.width);
  float overlapY = overlap1D(bbox1.top, bbox1.top + bbox1.height, bbox2.top, bbox2.top + bbox2.height);
  float area1 = (bbox1.width) * (bbox1.height);
  float area2 = (bbox2.width) * (bbox2.height);
  float overlap2D = overlapX * overlapY;
  float u = area1 + area2 - overlap2D;
  return u == 0 ? 0 : overlap2D / u;
}

static std::vector<NvDsInferInstanceMaskInfo>
nonMaximumSuppression(std::vector<NvDsInferInstanceMaskInfo> binfo)
{
  std::stable_sort(binfo.begin(), binfo.end(), [](const NvDsInferInstanceMaskInfo& b1,
      const NvDsInferInstanceMaskInfo& b2) {
    return b1.detectionConfidence > b2.detectionConfidence;
  });

  std::vector<NvDsInferInstanceMaskInfo> out;

  for (auto i : binfo) {
    bool keep = true;
    for (auto j : out) {
      if (keep) {
        float overlap = computeIoU(i, j);
        keep = overlap <= NMS_THRESH;
      }
      else {
        break;
      }
    }
    if (keep) {
      out.push_back(i);
    }
    else {
      delete[] i.mask;
    }
  }

  return out;
}

// The joints of one detection, as the pose module reads them: x, y and the confidence, in the
// network's input pixels, plus the network's input size.
//
// The joints are NOT clamped to the network's input, unlike the face parser's landmarks: a body
// joint can legitimately lie outside the crop (an arm outside the person's box), and the joint's
// own confidence is what says whether the network saw it.
static void
addKeypointsProposal(const float* detection, uint netW, uint netH, NvDsInferInstanceMaskInfo& b)
{
  b.mask = new float[KEYPOINT_VALUES];
  for (int k = 0; k < KEYPOINT_COUNT; ++k) {
    b.mask[k * KEYPOINT_DIMS + 0] = detection[k * KEYPOINT_DIMS + 0];
    b.mask[k * KEYPOINT_DIMS + 1] = detection[k * KEYPOINT_DIMS + 1];
    b.mask[k * KEYPOINT_DIMS + 2] = detection[k * KEYPOINT_DIMS + 2];
  }
  b.mask_width = netW;
  b.mask_height = netH;
  b.mask_size = sizeof(float) * KEYPOINT_VALUES;
}

// The detection's box, in the network's input pixels. False when the box is empty (a detection
// the network threw away), in which case the detection is not a person.
static bool
addBBoxProposal(float x1, float y1, float x2, float y2, uint netW, uint netH, float score,
    NvDsInferInstanceMaskInfo& b)
{
  x1 = clamp(x1, 0, netW);
  y1 = clamp(y1, 0, netH);
  x2 = clamp(x2, 0, netW);
  y2 = clamp(y2, 0, netH);

  b.left = x1;
  b.width = clamp(x2 - x1, 0, netW);
  b.top = y1;
  b.height = clamp(y2 - y1, 0, netH);

  if (b.width < 1 || b.height < 1) {
    return false;
  }

  b.detectionConfidence = score;
  b.classId = 0;                 // the pose model's one class: a person
  return true;
}

static bool
NvDsInferParseCustomYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  if (outputLayersInfo.empty()) {
    std::cerr << "ERROR - Could not find output layer" << std::endl;
    return false;
  }

  // The boxes layer decides the layout: its last axis is the 4 of a box either way, and the
  // number of axes says whether the batch axis is there.
  Layout layout{false, 0, 1};
  const NvDsInferLayerInfo* boxesLayer = findBoxesLayer(outputLayersInfo, layout);
  if (boxesLayer == nullptr) {
    std::cerr << "ERROR - the pose model must expose three outputs: boxes [N,4], scores [N] and "
              << "keypoints [N," << KEYPOINT_COUNT << "," << KEYPOINT_DIMS << "] (the batch axis "
              << "may be there too)" << std::endl;
    return false;
  }

  if (!layoutIsSane(layout)) {
    std::cerr << "ERROR - the pose model's boxes layer says " << layout.batch << " crop(s) of "
              << layout.count << " detection(s), which is not a real inference (a dim nvinfer "
              << "could not resolve comes as -1). The parser refuses to read it." << std::endl;
    return false;
  }

  const NvDsInferLayerInfo* scoresLayer =
      findLayer(outputLayersInfo, SCORES_LAYER, [layout](const NvDsInferLayerInfo& l) {
        return isScoresShape(l, layout);
      });
  const NvDsInferLayerInfo* keypointsLayer =
      findLayer(outputLayersInfo, KEYPOINTS_LAYER, [layout](const NvDsInferLayerInfo& l) {
        return isKeypointsShape(l, layout);
      });

  if (scoresLayer == nullptr || keypointsLayer == nullptr) {
    std::cerr << "ERROR - the pose model's outputs do not agree: the boxes say "
              << (layout.batched ? "a batch of " : "") << layout.count << " detection(s), so the "
              << "scores must be [" << (layout.batched ? "B,N" : "N") << "] and the keypoints ["
              << (layout.batched ? "B,N," : "N,") << KEYPOINT_COUNT << "," << KEYPOINT_DIMS << "]"
              << std::endl;
    return false;
  }

  // The buffers are read as float32. An engine that holds them in another precision would be read
  // as noise, so the parser refuses instead of reporting rubbish.
  for (const NvDsInferLayerInfo* layer : {boxesLayer, scoresLayer, keypointsLayer}) {
    if (layer->dataType != FLOAT) {
      std::cerr << "ERROR - the pose model's outputs must be float32; this layer's data type is "
                << layer->dataType << ". The engine was built at a precision the parser does not "
                << "read (network-mode=0 is fp32)." << std::endl;
      return false;
    }
  }

  const float* boxes = (const float*) (boxesLayer->buffer);
  const float* scores = (const float*) (scoresLayer->buffer);
  const float* keypoints = (const float*) (keypointsLayer->buffer);

  const size_t count = layout.count;
  const size_t batch = layout.batch;

  const float threshold = detectionParams.perClassPreclusterThreshold.empty()
      ? 0.0f : detectionParams.perClassPreclusterThreshold[0];

  std::vector<NvDsInferInstanceMaskInfo> objects;

  // One crop at a time: each crop is a DIFFERENT person, in its OWN network-input pixels, so the
  // NMS runs per crop -- an NMS across crops would suppress one person with another.
  for (size_t crop = 0; crop < batch; ++crop) {
    const float* cropBoxes = boxes + crop * count * 4;
    const float* cropScores = scores + crop * count;
    const float* cropKeypoints = keypoints + crop * count * KEYPOINT_VALUES;

    std::vector<NvDsInferInstanceMaskInfo> cropObjects;

    for (size_t n = 0; n < count; ++n) {
      const float score = cropScores[n];
      if (score < threshold) {
        continue;
      }

      const float* box = cropBoxes + n * 4;

      NvDsInferInstanceMaskInfo b;
      if (!addBBoxProposal(box[0], box[1], box[2], box[3], networkInfo.width, networkInfo.height,
              score, b)) {
        continue;
      }
      addKeypointsProposal(cropKeypoints + n * KEYPOINT_VALUES, networkInfo.width, networkInfo.height, b);

      cropObjects.push_back(b);
    }

    for (auto& o : nonMaximumSuppression(cropObjects)) {
      objects.push_back(o);
    }
  }

  objectList.clear();
  objectList = objects;

  return true;
}

extern "C" bool
NvDsInferParseYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo, NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams, std::vector<NvDsInferInstanceMaskInfo>& objectList)
{
  return NvDsInferParseCustomYoloPose(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloPose);
