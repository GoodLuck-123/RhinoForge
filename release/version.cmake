set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_LIST_DIR}/../VERSION"
    "${CMAKE_CURRENT_LIST_DIR}/runtime-dependencies.json")
file(READ "${CMAKE_CURRENT_LIST_DIR}/../VERSION" RHINOFORGE_VERSION)
string(STRIP "${RHINOFORGE_VERSION}" RHINOFORGE_VERSION)
if(NOT RHINOFORGE_VERSION MATCHES "^([0-9]+\\.[0-9]+\\.[0-9]+)(rc[0-9]+|\\.dev[0-9]+)?$")
    message(FATAL_ERROR "VERSION must contain a release, rc or dev version")
endif()
set(RHINOFORGE_CMAKE_VERSION "${CMAKE_MATCH_1}")
set(RHINOFORGE_VERSION_STAGE "${CMAKE_MATCH_2}")

file(READ "${CMAKE_CURRENT_LIST_DIR}/runtime-dependencies.json" RHINOFORGE_RUNTIME_DEPENDENCIES)
foreach(field IN ITEMS cmake_version release_version abi_version)
    string(JSON value GET "${RHINOFORGE_RUNTIME_DEPENDENCIES}" launch_kernel "${field}")
    string(TOUPPER "${field}" upper_field)
    set("RHINOFORGE_LAUNCH_${upper_field}" "${value}")
endforeach()
if(NOT RHINOFORGE_LAUNCH_RELEASE_VERSION STREQUAL RHINOFORGE_VERSION OR
   NOT RHINOFORGE_LAUNCH_CMAKE_VERSION STREQUAL RHINOFORGE_CMAKE_VERSION)
    message(FATAL_ERROR "RhinoForge and Rhino Launch must use the same release version")
endif()
