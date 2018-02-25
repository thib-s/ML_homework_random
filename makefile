
BIN_DIR=bin-${HOST}/
SRC_DIR=src/
TEST_DIR=opt/test/

all: ${BIN_DIR}${TEST_DIR}AbaloneTestStarcraft.class


${BIN_DIR}${TEST_DIR}AbaloneTestStarcraft.class : ${SRC_DIR}${TEST_DIR}AbaloneTestStarcraft.java ${BIN_DIR}
	javac -d ${BIN_DIR} -sourcepath ${SRC_DIR} ${SRC_DIR}${TEST_DIR}AbaloneTestStarcraft.java

${BIN_DIR}:
	mkdir ${BIN_DIR}

