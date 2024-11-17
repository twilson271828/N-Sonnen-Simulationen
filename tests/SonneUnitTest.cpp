#include "../include/sonne.hpp"
#include <gtest/gtest.h>

class SonneUnitTest : public ::testing::Test {
public:
  void SetUp() override {
    // long constructors


  }
  void TearDown() override {}

};

TEST_F(SonneUnitTest, ConstructorTests) {
    EXPECT_EQ(1, 1);
}


int main(int argc, char **argv) {

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

