#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hv_static" for configuration "Release"
set_property(TARGET hv_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hv_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C;CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libhv_static.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS hv_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_hv_static "${_IMPORT_PREFIX}/lib/libhv_static.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
