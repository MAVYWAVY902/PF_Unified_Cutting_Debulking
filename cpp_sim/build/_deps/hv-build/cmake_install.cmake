# Install script for directory: /home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/yunxin/pf-shape-matching/cpp_sim/build/lib/libhv_static.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/hv" TYPE FILE FILES
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/hv.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/hconfig.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/hexport.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hplatform.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hdef.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hatomic.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/herr.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/htime.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hmath.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hbase.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hversion.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hsysinfo.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hproc.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hthread.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hmutex.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hsocket.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hlog.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hbuf.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hmain.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/base/hendian.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/ssl/hssl.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/event/hloop.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/event/nlog.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/util/base64.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/util/md5.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/util/sha1.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hmap.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hstring.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hfile.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hpath.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hdir.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hurl.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hscope.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hthreadpool.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hasync.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/hobjectpool.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/ifconfig.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/iniparser.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/json.hpp"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/singleton.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/cpputil/ThreadLocalStorage.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/Buffer.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/Channel.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/Event.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/EventLoop.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/EventLoopThread.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/EventLoopThreadPool.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/Status.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/TcpClient.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/TcpServer.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/UdpClient.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/evpp/UdpServer.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/httpdef.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/wsdef.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/http_content.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/HttpMessage.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/HttpParser.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/WebSocketParser.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/WebSocketChannel.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/server/HttpServer.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/server/HttpService.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/server/HttpContext.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/server/HttpResponseWriter.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/server/WebSocketServer.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/client/HttpClient.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/client/requests.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/client/axios.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/client/AsyncHttpClient.h"
    "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-src/http/client/WebSocketClient.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/libhv/libhvConfig.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/libhv/libhvConfig.cmake"
         "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-build/CMakeFiles/Export/lib/cmake/libhv/libhvConfig.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/libhv/libhvConfig-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/libhv/libhvConfig.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/libhv" TYPE FILE FILES "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-build/CMakeFiles/Export/lib/cmake/libhv/libhvConfig.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/libhv" TYPE FILE FILES "/home/yunxin/pf-shape-matching/cpp_sim/build/_deps/hv-build/CMakeFiles/Export/lib/cmake/libhv/libhvConfig-release.cmake")
  endif()
endif()

