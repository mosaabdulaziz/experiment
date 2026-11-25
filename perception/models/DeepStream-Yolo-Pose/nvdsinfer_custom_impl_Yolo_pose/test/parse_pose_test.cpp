// The pose parser, proved on a plain host.
//
// The parser is C++ that normally runs inside the DeepStream container, where it is handed a
// TensorRT output buffer. This harness gives it the same three tensors by hand -- a tensor whose
// contents are known, which is what FALL_DETECTION_DESIGN.md §11 asks the parser to be proved
// against -- and checks what comes out:
//
//   - the threshold: a detection below the parser's own threshold is dropped;
//   - the NMS: an overlapping duplicate is dropped, the better one is kept;
//   - the payload: 17 joints of x, y and confidence, in the network's own pixels, plus the
//     network's input size, packed the way the pose module reads them;
//   - the box: in the network's pixels, clamped to the network's input;
//   - the layer lookup: the SHAPE is the requirement and the name only a preference, so a
//     renamed layer still resolves and a mis-shaped one is refused;
//   - the refusals: outputs that disagree on the number of detections, and a missing output.
//
// It prints one line per check and returns non-zero if any of them failed, so the pytest that
// drives it (tests/test_pose_parser.py) needs no parsing of the output.

#include "nvdsinfer_custom_impl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" bool
NvDsInferParseYoloPose(std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo, NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferInstanceMaskInfo>& objectList);

static const int KEYPOINT_COUNT = 17;
static const int KEYPOINT_DIMS = 3;
static const int KEYPOINT_VALUES = KEYPOINT_COUNT * KEYPOINT_DIMS;

static int failures = 0;

static void
check(bool ok, const std::string& what)
{
  if (ok) {
    printf("ok: %s\n", what.c_str());
  } else {
    printf("FAIL: %s\n", what.c_str());
    failures++;
  }
}

static void
check_near(float actual, float expected, float tolerance, const std::string& what)
{
  bool ok = std::fabs(actual - expected) <= tolerance;
  if (!ok) {
    printf("  (got %f, expected %f)\n", actual, expected);
  }
  check(ok, what);
}

// The three outputs, as the parser reads them. The EXPORT writes them with a batch axis
// (boxes [B,N,4], scores [B,N], keypoints [B,N,17,3]); nvinfer may hand the parser one crop's
// output with that axis stripped (boxes [N,4], scores [N], keypoints [N,17,3]). Both views are
// proved here.
struct Dump {
  std::vector<float> boxes;
  std::vector<float> scores;
  std::vector<float> keypoints;

  size_t count() const { return scores.size(); }
};

static NvDsInferLayerInfo
layer(const char* name, const std::vector<unsigned int>& dims, void* buffer)
{
  NvDsInferLayerInfo info;
  memset(&info, 0, sizeof(info));
  info.dataType = FLOAT;
  info.inferDims.numDims = (unsigned int) dims.size();
  info.inferDims.numElements = 1;
  for (size_t i = 0; i < dims.size() && i < NVDSINFER_MAX_DIMS; i++) {
    info.inferDims.d[i] = dims[i];
    info.inferDims.numElements *= dims[i];
  }
  info.bindingIndex = 0;
  info.layerName = name;
  info.buffer = buffer;
  info.isInput = 0;
  return info;
}

// The three layers, in one of the two views. `batch` is 1 for the per-crop view (the batch axis
// stripped) and the number of crops for the batched view (the shapes the export writes): the dump
// then holds `batch` crops of `count` detections each.
static std::vector<NvDsInferLayerInfo>
layers_of(const Dump& dump, const char* boxes_name, const char* scores_name,
    const char* keypoints_name, unsigned int batch = 1)
{
  const unsigned int count = batch == 1
      ? (unsigned int) dump.count() : (unsigned int) (dump.count() / batch);
  std::vector<NvDsInferLayerInfo> out;
  if (batch == 1) {
    out.push_back(layer(boxes_name, {count, 4}, (void*) dump.boxes.data()));
    out.push_back(layer(scores_name, {count}, (void*) dump.scores.data()));
    out.push_back(layer(keypoints_name, {count, (unsigned int) KEYPOINT_COUNT,
        (unsigned int) KEYPOINT_DIMS}, (void*) dump.keypoints.data()));
  } else {
    out.push_back(layer(boxes_name, {batch, count, 4}, (void*) dump.boxes.data()));
    out.push_back(layer(scores_name, {batch, count}, (void*) dump.scores.data()));
    out.push_back(layer(keypoints_name, {batch, count, (unsigned int) KEYPOINT_COUNT,
        (unsigned int) KEYPOINT_DIMS}, (void*) dump.keypoints.data()));
  }
  return out;
}

static NvDsInferNetworkInfo
network(unsigned int width, unsigned int height)
{
  NvDsInferNetworkInfo info;
  info.width = width;
  info.height = height;
  info.channels = 3;
  return info;
}

static void
free_masks(std::vector<NvDsInferInstanceMaskInfo>& objects)
{
  for (auto& o : objects) {
    delete[] o.mask;
    o.mask = nullptr;
  }
  objects.clear();
}

// The joints of one detection: joint k at (10 + k, 20 + k) with a confidence of 0.5 + k/100.
static void
fill_keypoints(Dump& dump, size_t n, float x_offset)
{
  for (int k = 0; k < KEYPOINT_COUNT; k++) {
    float* joint = &dump.keypoints[(n * KEYPOINT_COUNT + k) * KEYPOINT_DIMS];
    joint[0] = x_offset + (float) k;
    joint[1] = 20.0f + (float) k;
    joint[2] = 0.5f + (float) k / 100.0f;
  }
}

int
main()
{
  const float threshold = 0.25f;

  // ── the threshold, the payload and the box ────────────────────────────────────────────────────────────────
  {
    Dump dump;
    dump.boxes = {10.0f, 20.0f, 110.0f, 220.0f,      // detection 0: kept
                  300.0f, 400.0f, 400.0f, 500.0f};    // detection 1: below the threshold
    dump.scores = {0.90f, 0.10f};
    dump.keypoints.resize(2 * KEYPOINT_VALUES);
    fill_keypoints(dump, 0, 10.0f);
    fill_keypoints(dump, 1, 300.0f);

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold = {threshold};
    params.perClassPostclusterThreshold = {};

    auto layers = layers_of(dump, "boxes", "scores", "keypoints");
    std::vector<NvDsInferInstanceMaskInfo> objects;
    bool parsed = NvDsInferParseYoloPose(layers, network(640, 640), params, objects);

    check(parsed, "the parser accepts the three outputs");
    check(objects.size() == 1, "a detection below the threshold is dropped");

    if (objects.size() == 1) {
      const auto& o = objects[0];
      check_near(o.left, 10.0f, 0.001f, "the box's left edge is the network's pixels");
      check_near(o.top, 20.0f, 0.001f, "the box's top edge is the network's pixels");
      check_near(o.width, 100.0f, 0.001f, "the box's width is the network's pixels");
      check_near(o.height, 200.0f, 0.001f, "the box's height is the network's pixels");
      check_near(o.detectionConfidence, 0.90f, 0.001f, "the detection carries its score");
      check(o.classId == 0, "the pose model's one class is 0");
      check(o.mask_width == 640 && o.mask_height == 640, "the payload carries the network's size");
      check(o.mask_size == sizeof(float) * KEYPOINT_VALUES, "the payload is 17 x 3 floats");
      check(o.mask != nullptr, "the payload is allocated");

      if (o.mask != nullptr) {
        bool joints_ok = true;
        for (int k = 0; k < KEYPOINT_COUNT; k++) {
          joints_ok = joints_ok
              && std::fabs(o.mask[k * 3 + 0] - (10.0f + (float) k)) < 0.001f
              && std::fabs(o.mask[k * 3 + 1] - (20.0f + (float) k)) < 0.001f
              && std::fabs(o.mask[k * 3 + 2] - (0.5f + (float) k / 100.0f)) < 0.001f;
        }
        check(joints_ok, "the payload holds the 17 joints' x, y and confidence");
      }
    }
    free_masks(objects);
  }

  // ── the NMS: an overlapping duplicate is dropped, the better one is kept ────────────────────────────────
  {
    Dump dump;
    dump.boxes = {10.0f, 20.0f, 110.0f, 220.0f,       // detection 0: the better one
                  12.0f, 22.0f, 112.0f, 222.0f,       // detection 1: the same person, nearly
                  400.0f, 400.0f, 500.0f, 500.0f};     // detection 2: a different person
    dump.scores = {0.90f, 0.80f, 0.70f};
    dump.keypoints.resize(3 * KEYPOINT_VALUES);
    fill_keypoints(dump, 0, 10.0f);
    fill_keypoints(dump, 1, 12.0f);
    fill_keypoints(dump, 2, 400.0f);

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold = {threshold};
    params.perClassPostclusterThreshold = {};

    auto layers = layers_of(dump, "boxes", "scores", "keypoints");
    std::vector<NvDsInferInstanceMaskInfo> objects;
    bool parsed = NvDsInferParseYoloPose(layers, network(640, 640), params, objects);

    check(parsed, "the parser accepts three overlapping detections");
    check(objects.size() == 2, "the NMS drops the duplicate and keeps the other person");
    if (objects.size() == 2) {
      check_near(objects[0].detectionConfidence, 0.90f, 0.001f, "the better detection is kept");
      check_near(objects[1].detectionConfidence, 0.70f, 0.001f, "the other person is kept");
    }
    free_masks(objects);
  }

  // ── a joint outside the network's input keeps its value (the face parser clamps, this one does not) ────
  {
    Dump dump;
    dump.boxes = {10.0f, 20.0f, 110.0f, 220.0f};
    dump.scores = {0.90f};
    dump.keypoints.resize(KEYPOINT_VALUES);
    fill_keypoints(dump, 0, 10.0f);
    dump.keypoints[0] = -15.0f;          // a joint to the left of the crop
    dump.keypoints[1] = 900.0f;          // a joint below the crop

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold = {threshold};
    params.perClassPostclusterThreshold = {};

    auto layers = layers_of(dump, "boxes", "scores", "keypoints");
    std::vector<NvDsInferInstanceMaskInfo> objects;
    NvDsInferParseYoloPose(layers, network(640, 640), params, objects);

    check(objects.size() == 1, "the detection is kept");
    if (objects.size() == 1 && objects[0].mask != nullptr) {
      check_near(objects[0].mask[0], -15.0f, 0.001f, "a joint outside the crop keeps its x");
      check_near(objects[0].mask[1], 900.0f, 0.001f, "a joint outside the crop keeps its y");
    }
    free_masks(objects);
  }

  // ── the box is clamped to the network's input ───────────────────────────────────────────────────────────
  {
    Dump dump;
    dump.boxes = {-50.0f, -20.0f, 900.0f, 1200.0f};
    dump.scores = {0.90f};
    dump.keypoints.resize(KEYPOINT_VALUES);
    fill_keypoints(dump, 0, 10.0f);

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold = {threshold};
    params.perClassPostclusterThreshold = {};

    auto layers = layers_of(dump, "boxes", "scores", "keypoints");
    std::vector<NvDsInferInstanceMaskInfo> objects;
    NvDsInferParseYoloPose(layers, network(640, 640), params, objects);

    check(objects.size() == 1, "a box that runs past the network's input is kept");
    if (objects.size() == 1) {
      check_near(objects[0].left, 0.0f, 0.001f, "the box's left edge is clamped to the network");
      check_near(objects[0].top, 0.0f, 0.001f, "the box's top edge is clamped to the network");
      check_near(objects[0].width, 640.0f, 0.001f, "the box's width is clamped to the network");
      check_near(objects[0].height, 640.0f, 0.001f, "the box's height is clamped to the network");
    }
    free_masks(objects);
  }

  // ── the shape is the requirement, the name only a preference ───────────────────────────────────────────
  {
    Dump dump;
    dump.boxes = {10.0f, 20.0f, 110.0f, 220.0f};
    dump.scores = {0.90f};
    dump.keypoints.resize(KEYPOINT_VALUES);
    fill_keypoints(dump, 0, 10.0f);

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold = {threshold};
    params.perClassPostclusterThreshold = {};

    auto renamed = layers_of(dump, "output_0", "output_1", "output_2");
    std::vector<NvDsInferInstanceMaskInfo> objects;
    bool parsed = NvDsInferParseYoloPose(renamed, network(640, 640), params, objects);
    check(parsed && objects.size() == 1, "a renamed layer is still found by its shape");
    free_masks(objects);
  }

  // ── the NAME is how the boxes layer is found; the shape is the fallback ─────────────────────────────────
  {
    Dump dump;
    dump.boxes = {10.0f, 20.0f, 110.0f, 220.0f};
    dump.scores = {0.90f};
    dump.keypoints.resize(KEYPOINT_VALUES);
    fill_keypoints(dump, 0, 10.0f);

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold = {threshold};
    params.perClassPostclusterThreshold = {};

    // a decoy with the SAME shape comes first: the parser must read the layer the export NAMED,
    // not the first one whose shape happens to fit
    std::vector<float> decoy = {500.0f, 500.0f, 600.0f, 700.0f};
    auto layers = layers_of(dump, "boxes", "scores", "keypoints");
    layers.insert(layers.begin(), layer("output_0", {1, 4}, (void*) decoy.data()));

    std::vector<NvDsInferInstanceMaskInfo> objects;
    bool parsed = NvDsInferParseYoloPose(layers, network(640, 640), params, objects);
    check(parsed && objects.size() == 1, "a decoy layer with the same shape does not confuse the parser");
    if (objects.size() == 1) {
      check_near(objects[0].left, 10.0f, 0.001f, "the boxes layer the export NAMED is the one read");
    }
    free_masks(objects);
  }

  // ── the outputs as the EXPORT writes them: with the batch axis ─────────────────────────────────────────
  {
    // Two crops, each with ONE detection. The boxes overlap on purpose: each crop is a DIFFERENT
    // person, in its OWN network-input pixels, so the NMS must not run across the batch.
    Dump dump;
    dump.boxes = {10.0f, 20.0f, 110.0f, 220.0f,       // crop 0: one person
                  12.0f, 22.0f, 112.0f, 222.0f};     // crop 1: another person, in the same place
    dump.scores = {0.90f, 0.80f};
    dump.keypoints.resize(2 * KEYPOINT_VALUES);
    fill_keypoints(dump, 0, 10.0f);
    fill_keypoints(dump, 1, 500.0f);

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold = {threshold};
    params.perClassPostclusterThreshold = {};

    // boxes [B,N,4], scores [B,N], keypoints [B,N,17,3] -- the shapes the export writes
    auto layers = layers_of(dump, "boxes", "scores", "keypoints", 2);
    std::vector<NvDsInferInstanceMaskInfo> objects;
    bool parsed = NvDsInferParseYoloPose(layers, network(640, 640), params, objects);

    check(parsed, "the parser accepts the outputs with the batch axis");
    check(objects.size() == 2, "both crops' detections are read");
    if (objects.size() == 2) {
      check_near(objects[0].left, 10.0f, 0.001f, "the first crop's box is the first person's");
      check_near(objects[1].left, 12.0f, 0.001f, "the second crop's box is the second person's");
      check_near(objects[0].detectionConfidence, 0.90f, 0.001f, "the first crop carries its own score");
      check_near(objects[1].detectionConfidence, 0.80f, 0.001f, "the second crop carries its own score");
      if (objects[0].mask != nullptr && objects[1].mask != nullptr) {
        check_near(objects[0].mask[0], 10.0f, 0.001f, "the first crop carries its own joints");
        check_near(objects[1].mask[0], 500.0f, 0.001f, "the second crop carries its own joints");
      }
    }
    free_masks(objects);
  }

  // ── the refusals ────────────────────────────────────────────────────────────────────────────────────────
  {
    Dump dump;
    dump.boxes = {10.0f, 20.0f, 110.0f, 220.0f};
    dump.scores = {0.90f};
    dump.keypoints.resize(KEYPOINT_VALUES);
    fill_keypoints(dump, 0, 10.0f);

    NvDsInferParseDetectionParams params;
    params.numClassesConfigured = 1;
    params.perClassPreclusterThreshold = {threshold};
    params.perClassPostclusterThreshold = {};

    // an output with fewer detections than the boxes: reading it would run off the end
    {
      auto layers = layers_of(dump, "boxes", "scores", "keypoints");
      layers[1].inferDims.d[0] = 0;
      std::vector<NvDsInferInstanceMaskInfo> objects;
      bool parsed = NvDsInferParseYoloPose(layers, network(640, 640), params, objects);
      check(!parsed, "an output with fewer detections than the boxes is refused");
      free_masks(objects);
    }
    // an output with the wrong shape
    {
      auto layers = layers_of(dump, "boxes", "scores", "keypoints");
      layers[0].inferDims.d[1] = 5;
      std::vector<NvDsInferInstanceMaskInfo> objects;
      bool parsed = NvDsInferParseYoloPose(layers, network(640, 640), params, objects);
      check(!parsed, "an output with the wrong shape is refused");
      free_masks(objects);
    }
    // an output that is missing
    {
      auto layers = layers_of(dump, "boxes", "scores", "keypoints");
      layers.pop_back();
      std::vector<NvDsInferInstanceMaskInfo> objects;
      bool parsed = NvDsInferParseYoloPose(layers, network(640, 640), params, objects);
      check(!parsed, "a missing output is refused");
      free_masks(objects);
    }
    // an output that is not float32: read as float, it would be noise
    {
      auto layers = layers_of(dump, "boxes", "scores", "keypoints");
      layers[2].dataType = HALF;
      std::vector<NvDsInferInstanceMaskInfo> objects;
      bool parsed = NvDsInferParseYoloPose(layers, network(640, 640), params, objects);
      check(!parsed, "an output that is not float32 is refused");
      free_masks(objects);
    }
    // a dim nvinfer could not resolve: -1, which as an unsigned count is 4294967295. The crop loop
    // would walk four billion crops off the end of the buffer.
    {
      Dump two;
      two.boxes = {10.0f, 20.0f, 110.0f, 220.0f, 12.0f, 22.0f, 112.0f, 222.0f};
      two.scores = {0.90f, 0.80f};
      two.keypoints.resize(2 * KEYPOINT_VALUES);
      fill_keypoints(two, 0, 10.0f);
      fill_keypoints(two, 1, 500.0f);

      auto layers = layers_of(two, "boxes", "scores", "keypoints", 2);
      layers[0].inferDims.d[0] = (unsigned int) -1;
      std::vector<NvDsInferInstanceMaskInfo> objects;
      bool parsed = NvDsInferParseYoloPose(layers, network(640, 640), params, objects);
      check(!parsed, "a batch dim nvinfer could not resolve is refused");
      free_masks(objects);
    }
  }

  if (failures > 0) {
    printf("%d check(s) failed\n", failures);
    return 1;
  }
  printf("all checks passed\n");
  return 0;
}
