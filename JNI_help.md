# JNI Help Guide

JNI...................
It is not as easy as it seems to be. I have wasted hours on this, but let me save yours.

Read this article first if you've no knowledge on JNI:
https://medium.com/@oleksandra_shershen/harnessing-the-power-of-jni-seamless-integration-of-java-and-c-for-enhanced-performance-d1a6ac4f4a77
(If you want no knowledge, skip to the commands)

I hope you've read it. I hope you understand how JNI works.

I have put all the classes and libs in build/classes and build/libs in the root directory. And I have separated the java and cpp code by giving each a separate folder src/java and src/cpp, this way our project stays structured.

Now the first step is to setup your Java JDK and its Home, using JAVA_HOME, i've given the commands for that.

Then we compile all our Java and cpp files, in build/classes

Then we compile our cpp lib in build/library

Then we run it. 

## Our Project Structure
```
ML-LIBRARY/
├─ build/
│  ├─ classes/
│  └─ libs/
├─ src/
│  ├─ cpp/
│  ├─ gui/
│  └─ java/
```

## Sequence of commands from Root Directory (Ml-Library)

### Linux

**Step 1: Install JDK 24**

```bash
sudo apt update
sudo apt install openjdk-24-jdk
java -version
javac -version
```

**Step 2: Set JAVA_HOME**

```bash
export JAVA_HOME=/usr/lib/jvm/java-24-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
```

Make permanent:

```bash
echo 'export JAVA_HOME=/usr/lib/jvm/java-24-openjdk-amd64' >> ~/.bashrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

**Step 3: Compile all Java Files**

```bash
mkdir -p build/classes

javac \
-d build/classes \
-h src/cpp \
src/java/core/utils/matrix/MatrixOperations.java \
src/java/Main.java
```

**Step 4: Compile the cpp and build library**

Note: library name = "libmatrixops.so" it should not change it is hardcoded

```bash
mkdir -p build/libs

g++ -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/linux" \
    -shared -fPIC src/cpp/MatrixOperations.cpp \
    -o build/libs/libmatrixops.so
```

**Step 5: Run**

```bash
java -cp build/classes -Djava.library.path=build/libs Main
```

### Mac

**Step 1: Install JDK 24**

```bash
brew install openjdk@24
```

**Step 2: Set JAVA_HOME**

```bash
export JAVA_HOME=$(/usr/libexec/java_home -v 24)
export PATH=$JAVA_HOME/bin:$PATH
```

Make permanent:

```bash
echo 'export JAVA_HOME=$(/usr/libexec/java_home -v 24)' >> ~/.zshrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc
source ~/.zshrc
```

**Step 3: Compile all Java Files**

```bash
mkdir -p build/classes

javac \
-d build/classes \
-h src/cpp \
src/java/core/utils/matrix/MatrixOperations.java \
src/java/Main.java
```

**Step 4: Compile the cpp and build library**

Note: library name = "libmatrixops.dylib" it should not change it is hardcoded

```bash
mkdir -p build/libs

g++ -I"$JAVA_HOME/include" -I"$JAVA_HOME/include/darwin" \
-dynamiclib src/cpp/MatrixOperations.cpp \
-o build/libs/libmatrixops.dylib
```

**Step 5: Run**

```bash
java -cp build/classes -Djava.library.path=build/libs Main
```

### Windows

**Step 1: Install JDK 24**

- Download from Oracle JDK 24 and install.
- Verify:

```cmd
java -version
javac -version
```

**Step 2: Set JAVA_HOME**

```cmd
setx JAVA_HOME "C:\Program Files\Java\jdk-24"
setx PATH "%JAVA_HOME%\bin;%PATH%"
```

**Step 3: Compile all Java Files**

```cmd
mkdir build\classes

javac ^
-d build\classes ^
-h src\cpp ^
src\java\core\utils\matrix\MatrixOperations.java ^
src\java\Main.java
```

**Step 4: Compile the cpp and build library**

Note: library name = "libmatrixops.dll" it should not change it is hardcoded

```cmd
mkdir build\libs

g++ -I"%JAVA_HOME%\include" -I"%JAVA_HOME%\include\win32" ^
-shared src\cpp\MatrixOperations.cpp ^
-o build\libs\libmatrixops.dll
```

**Step 5: Run**

```cmd
java -cp build\classes -Djava.library.path=build\libs Main
```

## Troubleshooting Common Issues

### Library Loading Issues
- Make sure the library name matches exactly what's hardcoded in your Java code
- Check that the library file extension is correct for your platform (.so for Linux, .dylib for Mac, .dll for Windows)
- Verify that the `-Djava.library.path` points to the correct directory

### Compilation Errors
- Ensure JAVA_HOME is set correctly and points to JDK (not JRE)
- Check that all include paths are correct in the g++ command
- Make sure all Java files compile without errors before attempting to compile the native library

### Runtime Errors
- Verify that the native method signatures in your C++ code match exactly with the Java declarations
- Check that you're loading the library correctly in your Java code using `System.loadLibrary()`
- Ensure the library is in the correct path specified by `-Djava.library.path`