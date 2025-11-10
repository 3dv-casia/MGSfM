{
  description = "NixOS derivation of MGSfM (Multi-View Geometry and Structure from Motion)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/release-25.05";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.permittedInsecurePackages = [
            # Colmap dependency
            "freeimage-3.18.0-unstable-2024-04-18"
          ];
        };

        # --------------------------------------------------------------------
        # 1. PoseLib derivation – built from the upstream GitHub repo.
        #    We use fetchFromGitHub + buildCmakePackage (the library has its own CMakeLists.txt).
        # --------------------------------------------------------------------
        poselib = pkgs.stdenv.mkDerivation rec {
          pname   = "poselib";
          version = "v2.0.5";
          src     = pkgs.fetchFromGitHub {
            owner  = "PoseLib";
            repo   = "PoseLib";
            rev    = "${version}";
            sha256 = "sha256-fARRKT2UoPuuk9FOOsBdrACwGiGXWg/mLV4R0QIjeak=";
          };

          nativeBuildInputs = with pkgs; [ cmake ninja  ];
          buildInputs       = [ pkgs.eigen ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DBUILD_SHARED_LIBS=ON"   # expose as shared lib for MGSfM
          ];

          doCheck = false;
        };

        colmap = pkgs.stdenv.mkDerivation rec {
          pname   = "colmap";
          version = "3.11.1";
          src     = pkgs.fetchFromGitHub {
            owner  = "colmap";
            repo   = "colmap";
            rev    = "682ea9ac4020a143047758739259b3ff04dabe8d";
            sha256 = "sha256-xtA0lEAq38/AHI3C9FhvjV5JPfVawrFr1fga4J1pi/0=";
          };

          nativeBuildInputs = with pkgs; [ cmake ninja qt5.wrapQtAppsHook ];
          buildInputs       = [
            pkgs.boost
            pkgs.eigen
            pkgs.sqlite
            pkgs.libGL
            pkgs.ceres-solver
            pkgs.cgal
            pkgs.gmp
            pkgs.mpfr
            pkgs.qt5Full
            pkgs.flann
            pkgs.glew
            pkgs.freeimage
            poselib
          ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DBUILD_SHARED_LIBS=ON"   # expose as shared lib for MGSfM
            "-DFETCH_POSELIB=OFF"    # we provide poselib via buildInputs
          ];

          doCheck = false;
        };

        # --------------------------------------------------------------------
        # 2. MGSfM derivation – pulls from GitHub and links against COLMAP & PoseLib.
        # --------------------------------------------------------------------
        mgsfm = pkgs.stdenv.mkDerivation rec {
          pname   = "mgsfm";
          version = "1.0.0";   # matches the upstream project’s version tag
          src     = pkgs.fetchFromGitHub {
            owner  = "3dv-casia";
            repo   = "MGSfM";
            rev    = "master";                # or a specific commit/tag
            sha256 = "sha256-ekpY4PcdCP2W6+aFjsoXta1EUiqu5AHXhwMwjZ9Fqds=";
          };

          nativeBuildInputs = [ pkgs.cmake pkgs.ninja ];
          buildInputs       = [
            pkgs.eigen
            pkgs.blas
            pkgs.suitesparse
            pkgs.ceres-solver
            pkgs.boost
            pkgs.flann
            pkgs.sqlite
            pkgs.libGL
            pkgs.cgal
            pkgs.gmp
            pkgs.mpfr
            pkgs.qt5Full
            pkgs.glew
            pkgs.freeimage
            poselib # built PoseLib
            colmap  # build colmap
          ];

          cmakeFlags = [
            "-DCMAKE_BUILD_TYPE=Release"
            "-DFETCH_POSELIB=OFF"    # we provide poselib via buildInputs
            "-DFETCH_COLMAP=OFF"   # we provide COLMAP via buildInputs
          ];

          doCheck = false;  # the repo’s tests are disabled by default (TESTS_ENABLED OFF)
          installPhase = ''
            mkdir -p $out/bin
            cp glomap/mgsfm $out/bin/
            cp glomap/libglomap.a $out/bin/
            cp glomap/glomap $out/bin/
          '';
        };
      in {
        packages.default = [ mgsfm colmap ];
        devShells.default = pkgs.mkShell {
          buildInputs = [ mgsfm colmap ];
        };
      }
    );
}
