
BIN_DIR=bin-${HOST}/
SRC_DIR=src/
TEST_DIR=opt/test/

all: ${BIN_DIR}${TEST_DIR}TravelingSalesmanProblem.class ${BIN_DIR}${TEST_DIR}OptimizationTest.class


${BIN_DIR}${TEST_DIR}TravelingSalesmanProblem.class : ${SRC_DIR}${TEST_DIR}TravelingSalesmanProblem.java ${BIN_DIR}
	javac -d ${BIN_DIR} -sourcepath ${SRC_DIR} ${SRC_DIR}${TEST_DIR}TravelingSalesmanProblem.java

${BIN_DIR}${TEST_DIR}OptimizationTest.class : ${SRC_DIR}${TEST_DIR}OptimizationTest.java ${BIN_DIR}
	javac -d ${BIN_DIR} -sourcepath ${SRC_DIR} ${SRC_DIR}${TEST_DIR}OptimizationTest.java


${BIN_DIR}:
	mkdir ${BIN_DIR}

