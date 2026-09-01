# Release tools embed EMBER_CONFIGURED_GIT_HEAD in their evidence. CMake cache
# values survive checkout changes, so validate the live source revision every
# time one of those tools is built. Exported source archives deliberately have
# no .git entry; their caller-supplied revision remains the only available
# binding and is validated syntactically by the main configure step.

if(NOT DEFINED EMBER_EXPECTED_GIT_HEAD OR
   NOT EMBER_EXPECTED_GIT_HEAD MATCHES "^[0-9a-f][0-9a-f]*$")
  message(FATAL_ERROR "invalid expected Ember git revision")
endif()
string(LENGTH "${EMBER_EXPECTED_GIT_HEAD}" EMBER_EXPECTED_GIT_HEAD_LENGTH)
if(NOT EMBER_EXPECTED_GIT_HEAD_LENGTH EQUAL 40)
  message(FATAL_ERROR "expected Ember git revision must be 40 hex characters")
endif()

if(NOT EXISTS "${EMBER_SOURCE_DIR}/.git")
  return()
endif()

if(NOT EMBER_GIT_EXECUTABLE)
  find_program(EMBER_GIT_EXECUTABLE git)
endif()
if(NOT EMBER_GIT_EXECUTABLE)
  message(FATAL_ERROR
    "cannot verify Ember git revision: git metadata exists but git is unavailable")
endif()

execute_process(
  # Container builds run as root against a user-owned bind mount. Trust only
  # this exact source path for the read-only revision query rather than
  # requiring a persistent global safe.directory mutation in the image.
  COMMAND "${EMBER_GIT_EXECUTABLE}"
    -c "safe.directory=${EMBER_SOURCE_DIR}"
    -C "${EMBER_SOURCE_DIR}" rev-parse HEAD
  RESULT_VARIABLE EMBER_GIT_RESULT
  OUTPUT_VARIABLE EMBER_ACTUAL_GIT_HEAD
  ERROR_VARIABLE EMBER_GIT_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT EMBER_GIT_RESULT EQUAL 0)
  string(STRIP "${EMBER_GIT_ERROR}" EMBER_GIT_ERROR)
  message(FATAL_ERROR
    "cannot verify Ember git revision in ${EMBER_SOURCE_DIR}: ${EMBER_GIT_ERROR}")
endif()

if(NOT EMBER_ACTUAL_GIT_HEAD STREQUAL EMBER_EXPECTED_GIT_HEAD)
  message(FATAL_ERROR
    "stale Ember build directory: configured revision "
    "${EMBER_EXPECTED_GIT_HEAD}, source HEAD ${EMBER_ACTUAL_GIT_HEAD}; "
    "reconfigure before building release tools")
endif()
