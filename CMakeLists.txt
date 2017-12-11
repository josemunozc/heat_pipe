CMAKE_MINIMUM_REQUIRED(VERSION 2.8.8)

FIND_PACKAGE(deal.II 8.4.1 REQUIRED
  HINTS /opt/dealii/
  )
DEAL_II_INITIALIZE_CACHED_VARIABLES()

PROJECT(ihts)

INCLUDE_DIRECTORIES(/mnt/c/Users/Jose/Documents/code/git/libraries)

ADD_LIBRARY(mylib /mnt/c/Users/Jose/Documents/code/git/libraries/SurfaceCoefficients.cpp /mnt/c/Users/Jose/Documents/code/git/libraries/AnalyticSolution.cpp /mnt/c/Users/Jose/Documents/code/git/libraries/MaterialData.cpp /mnt/c/Users/Jose/Documents/code/git/libraries/BoundaryConditions.cpp /mnt/c/Users/Jose/Documents/code/git/libraries/DataTools.cpp /mnt/c/Users/Jose/Documents/code/git/libraries/Names.cpp)
DEAL_II_SETUP_TARGET(mylib)


ADD_EXECUTABLE(mycode pipe_3d.cc)
DEAL_II_SETUP_TARGET(mycode)

TARGET_LINK_LIBRARIES(mycode mylib)


ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )
