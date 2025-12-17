#include "engine/computeEngine.cuh"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <vector>

using T = cuFloatComplex;

class MockComputeEngine : public ComputeEngine<T> {
public:
  MockComputeEngine(const Params &p) : ComputeEngine<T>(p) {}

  MOCK_METHOD(void, solveStep, (int t), (override));
  MOCK_METHOD(int, getDownloadFrequency, (), (override));
  MOCK_METHOD(int, getTotalSteps, (), (override));
  MOCK_METHOD(void, appendFrame, (std::vector<cuFloatComplex> &), (override));
  MOCK_METHOD(void, saveResults, (const std::string &), (override));

  int getDownloadIterator() const { return downloadIterator; }
  const std::vector<T> &getHistoryData() const { return historyData; }
};

using ::testing::_;
using ::testing::AtLeast;
using ::testing::Invoke;
using ::testing::Return;

class ComputeEngineTest : public ::testing::Test {
protected:
  Params dummyParams = {.output = "output.json",
                        .simulationMode = SimulationMode::GrossPitaevskii,

                        .test = {.iterations = 8192,
                                 .gridWidth = 512,
                                 .gridHeight = 512,
                                 .threadsPerBlockX = 32,
                                 .threadsPerBlockY = 32,
                                 .downloadFrequency = 32},

                        .grossPitaevskii = {.iterations = 8192,
                                            .gridWidth = 512,
                                            .gridHeight = 512,
                                            .threadsPerBlockX = 32,
                                            .threadsPerBlockY = 32,
                                            .downloadFrequency = 32,
                                            .L = 1.0f,
                                            .sigma = 0.1f,
                                            .x0 = 0.15f,
                                            .y0 = 0.15f,
                                            .kx = 0.0f,
                                            .ky = 0.0f,
                                            .amp = 1.0f,
                                            .omega = 0.0f,
                                            .trapStr = 10e4f,
                                            .dt = 6e-7f,
                                            .g = 10e1f,
                                            .V_bias = 10.0f,
                                            .r_0 = 0.05f,
                                            .sigma2 = 0.025f,
                                            .absorbStrength = 10e3f,
                                            .absorbWidth = 0.025f}};
};

TEST_F(ComputeEngineTest, RunExecutesCorrectNumberOfSteps) {
  MockComputeEngine engine(dummyParams);

  int totalSteps = 21;

  EXPECT_CALL(engine, getTotalSteps()).WillRepeatedly(Return(totalSteps));

  EXPECT_CALL(engine, solveStep(_)).Times(totalSteps);

  EXPECT_CALL(engine, getDownloadFrequency()).WillRepeatedly(Return(10));

  EXPECT_CALL(engine, appendFrame(_)).Times(3);

  engine.run();
}

TEST_F(ComputeEngineTest, AppendsFrameBasedOnFrequency) {
  MockComputeEngine engine(dummyParams);

  int totalSteps = 5;
  int frequency = 2;

  EXPECT_CALL(engine, getTotalSteps()).WillRepeatedly(Return(totalSteps));
  EXPECT_CALL(engine, getDownloadFrequency()).WillRepeatedly(Return(frequency));

  EXPECT_CALL(engine, solveStep(_)).Times(totalSteps);

  EXPECT_CALL(engine, appendFrame(_)).Times(3);

  engine.run();
}

TEST_F(ComputeEngineTest, HistoryDataIsPassedByReference) {
  MockComputeEngine engine(dummyParams);

  EXPECT_CALL(engine, getTotalSteps()).WillRepeatedly(Return(1));
  EXPECT_CALL(engine, getDownloadFrequency()).WillRepeatedly(Return(1));
  EXPECT_CALL(engine, solveStep(_));

  EXPECT_CALL(engine, appendFrame(_))
      .WillOnce(Invoke([](std::vector<cuFloatComplex> &history) {
        cuFloatComplex val;
        val.x = 1.0f;
        val.y = 2.0f;
        history.push_back(val);
      }));

  engine.run();

  auto history = engine.getHistoryData();
  ASSERT_EQ(history.size(), 1);
  EXPECT_FLOAT_EQ(history[0].x, 1.0f);
}
