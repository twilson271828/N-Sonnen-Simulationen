#include "../include/Sonne.hpp"
#include <gtest/gtest.h>

class SonneUnitTest : public ::testing::Test {
public:
  void SetUp() override {
    // long constructors


  }
  void TearDown() override {}

};

TEST_F(SonneTest, ConstructorTests) {
    EXPECT_EQ(1, 1);
}