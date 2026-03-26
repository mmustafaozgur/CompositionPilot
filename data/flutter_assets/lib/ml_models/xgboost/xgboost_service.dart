import 'dart:async';
import 'dart:io';
import 'dart:convert';

class XGBoostService {
  static const String _baseUrl = 'http://localhost:5000';
  static const String _setupScript = 'setup.py';
  static const String _serviceScript = 'xgboost_service.py';

  // Create a safe UTF-8 decoder that allows malformed input
  static const _safeUtf8Decoder = Utf8Decoder(allowMalformed: true);

  Process? _process;
  bool _isRunning = false;
  bool _isSetupComplete = false;
  final _setupProgressController = StreamController<String>.broadcast();
  final _setupErrorController = StreamController<String>.broadcast();

  Stream<String> get setupProgress => _setupProgressController.stream;
  Stream<String> get setupError => _setupErrorController.stream;
  bool get isRunning => _isRunning;
  bool get isSetupComplete => _isSetupComplete;

  Future<String> _getPythonExecutable() async {
    if (Platform.isWindows) {
      // Windows-specific Python detection
      final commonWindowsPaths = [
        'python.exe',
        'py.exe',
        'C:\\Python311\\python.exe',
        'C:\\Python310\\python.exe',
        'C:\\Python39\\python.exe',
        'C:\\Python38\\python.exe',
        'C:\\Python37\\python.exe',
      ];

      // Try using 'where' command to find python
      try {
        final result = await Process.run('where', ['python'], runInShell: true);
        if (result.exitCode == 0 &&
            result.stdout.toString().trim().isNotEmpty) {
          final path = result.stdout.toString().trim().split('\n')[0];
          _setupProgressController
              .add('Found Python via "where" command: $path');

          // Verify it works
          final testResult = await Process.run(path, ['--version']);
          if (testResult.exitCode == 0) {
            return path;
          }
        }
      } catch (e) {
        _setupProgressController.add('Error running "where python": $e');
      }

      // Try using 'py' launcher
      try {
        final result = await Process.run('py', ['--version']);
        if (result.exitCode == 0) {
          _setupProgressController.add(
              'Found Python via py launcher: ${result.stdout.toString().trim()}');
          return 'py';
        }
      } catch (e) {
        _setupProgressController.add('Error running "py --version": $e');
      }

      // Try common Windows paths
      for (final path in commonWindowsPaths) {
        try {
          final result = await Process.run(path, ['--version']);
          if (result.exitCode == 0) {
            _setupProgressController.add(
                'Found Python at: $path (${result.stdout.toString().trim()})');
            return path;
          }
        } catch (e) {
          continue;
        }
      }

      throw Exception(
          'Python not found. Please install Python 3.7+ and add it to your system PATH.\n'
          'Tried: py launcher, where command, and common installation paths.');
    } else {
      // Unix/Linux/macOS
      return 'python3';
    }
  }

  Future<void> startService() async {
    if (_isRunning) return;

    try {
      _setupProgressController.add('Detecting Python executable...');

      // Get Python executable
      final pythonExecutable = await _getPythonExecutable();
      _setupProgressController.add('Python found: $pythonExecutable');

      // Get the absolute path to the script directory
      final scriptDir = Directory.current.path;
      final setupScriptPath =
          '$scriptDir${Platform.pathSeparator}lib${Platform.pathSeparator}ml_models${Platform.pathSeparator}xgboost${Platform.pathSeparator}$_setupScript';
      final serviceScriptPath =
          '$scriptDir${Platform.pathSeparator}lib${Platform.pathSeparator}ml_models${Platform.pathSeparator}xgboost${Platform.pathSeparator}$_serviceScript';
      final modelPath =
          '$scriptDir${Platform.pathSeparator}lib${Platform.pathSeparator}ml_models${Platform.pathSeparator}xgboost${Platform.pathSeparator}xgboost_trained_model.json';
      final columnsPath =
          '$scriptDir${Platform.pathSeparator}lib${Platform.pathSeparator}ml_models${Platform.pathSeparator}xgboost${Platform.pathSeparator}columns_data.csv';

      _setupProgressController.add('Checking required files...');

      // Verify all required files exist
      final requiredFiles = [
        setupScriptPath,
        serviceScriptPath,
        modelPath,
        columnsPath,
      ];

      for (final file in requiredFiles) {
        if (!File(file).existsSync()) {
          throw Exception('Required file not found: $file');
        }
        _setupProgressController
            .add('Found: ${file.split(Platform.pathSeparator).last}');
      }

      // Run setup script first
      _setupProgressController.add('Starting setup process...');
      _setupProgressController
          .add('Using Python executable: $pythonExecutable');
      _setupProgressController.add('Setup script path: $setupScriptPath');

      final setupProcess = await Process.start(
        pythonExecutable,
        [setupScriptPath],
        workingDirectory: scriptDir,
        runInShell: Platform.isWindows,
      );

      _setupProgressController
          .add('Setup process started, waiting for completion...');

      // Handle setup script output
      setupProcess.stdout.transform(_safeUtf8Decoder).listen((data) {
        final lines = data.trim().split('\n');
        for (final line in lines) {
          if (line.trim().isNotEmpty) {
            _setupProgressController.add('Setup: $line');
          }
        }
      });

      setupProcess.stderr.transform(_safeUtf8Decoder).listen((data) {
        final lines = data.trim().split('\n');
        for (final line in lines) {
          if (line.trim().isNotEmpty) {
            _setupErrorController.add('Setup Error: $line');
          }
        }
      });

      // Wait for setup to complete with timeout
      _setupProgressController
          .add('Waiting for setup to complete (max 5 minutes)...');

      final setupExitCode = await setupProcess.exitCode.timeout(
        const Duration(minutes: 5),
        onTimeout: () {
          _setupProgressController
              .add('Setup timeout reached, killing process...');
          setupProcess.kill();
          throw Exception('Setup timed out after 5 minutes');
        },
      );

      _setupProgressController
          .add('Setup process completed with exit code: $setupExitCode');

      if (setupExitCode != 0) {
        _setupErrorController.add('Setup failed with exit code $setupExitCode');
        return;
      }

      _setupProgressController.add('Setup completed successfully');
      _isSetupComplete = true;

      // Start the Flask service with environment variables
      _setupProgressController.add('Starting XGBoost Flask service...');
      _setupProgressController.add('Service script path: $serviceScriptPath');

      // Use virtual environment Python if it exists
      final venvDir =
          '$scriptDir${Platform.pathSeparator}lib${Platform.pathSeparator}ml_models${Platform.pathSeparator}xgboost${Platform.pathSeparator}venv';
      final venvPython = Platform.isWindows
          ? '$venvDir${Platform.pathSeparator}Scripts${Platform.pathSeparator}python.exe'
          : '$venvDir${Platform.pathSeparator}bin${Platform.pathSeparator}python';

      final servicePython =
          File(venvPython).existsSync() ? venvPython : pythonExecutable;
      _setupProgressController.add('Using Python for service: $servicePython');

      _setupProgressController.add('Setting up environment variables...');
      _setupProgressController.add('MODEL_PATH: $modelPath');
      _setupProgressController.add('COLUMNS_PATH: $columnsPath');

      _process = await Process.start(
        servicePython,
        [serviceScriptPath],
        workingDirectory: scriptDir,
        runInShell: Platform.isWindows,
        environment: {
          'MODEL_PATH': modelPath,
          'COLUMNS_PATH': columnsPath,
        },
      );

      _setupProgressController.add('Flask service process started');

      // Handle service output
      _process!.stdout.transform(_safeUtf8Decoder).listen((data) {
        final lines = data.trim().split('\n');
        for (final line in lines) {
          if (line.trim().isNotEmpty) {
            _setupProgressController.add('Service: $line');
          }
        }
      });

      _process!.stderr.transform(_safeUtf8Decoder).listen((data) {
        final lines = data.trim().split('\n');
        for (final line in lines) {
          if (line.trim().isNotEmpty) {
            _setupErrorController.add('Service Error: $line');
          }
        }
      });

      // Wait for service to be ready with timeout
      _setupProgressController.add('Waiting for service to be ready...');

      try {
        await _waitForService();
        _isRunning = true;
        _setupProgressController.add('XGBoost service is ready and running!');
      } catch (e) {
        _setupErrorController.add('Service failed to start: $e');
        await stopService();
        rethrow;
      }
    } catch (e) {
      _setupErrorController.add('Failed to start service: $e');
      _setupErrorController.add('Stack trace: ${StackTrace.current}');
      _isRunning = false;
      rethrow;
    }
  }

  Future<void> _waitForService() async {
    const maxAttempts = 30;
    var attempts = 0;

    _setupProgressController.add('Testing service connection...');

    while (attempts < maxAttempts) {
      try {
        final client = HttpClient();
        client.connectionTimeout = const Duration(seconds: 2);

        // Try to make a simple GET request to the root endpoint
        final request = await client.getUrl(Uri.parse('$_baseUrl/'));
        final response = await request.close();
        await response.drain();
        client.close();

        _setupProgressController.add('Service is responding on $_baseUrl');
        return;
      } catch (e) {
        attempts++;
        if (attempts <= 3 || attempts % 5 == 0) {
          _setupProgressController.add(
              'Connection attempt $attempts/$maxAttempts (${e.toString().length > 50 ? "${e.toString().substring(0, 50)}..." : e.toString()})');
        }

        // Give a bit more time for the service to initialize
        await Future.delayed(const Duration(seconds: 2));
      }
    }

    throw Exception(
        'Service failed to start within ${maxAttempts * 2} seconds. No response from $_baseUrl');
  }

  Future<void> stopService() async {
    if (!_isRunning) return;

    try {
      _process?.kill();
      await _process?.exitCode;
      _isRunning = false;
    } catch (e) {
      _setupErrorController.add('Error stopping service: $e');
    }
  }

  Future<Map<String, dynamic>> runModel(
      List<String> elements, int iterations) async {
    if (!_isRunning) {
      throw Exception('Service is not running');
    }

    try {
      final client = HttpClient();
      final request = await client.postUrl(Uri.parse('$_baseUrl/run_model'));
      request.headers.set('Content-Type', 'application/json');
      request.write(jsonEncode({
        'elements': elements,
        'iterations': iterations,
      }));

      final response = await request.close();
      final responseBody = await response.transform(_safeUtf8Decoder).join();
      final result = jsonDecode(responseBody);

      if (response.statusCode != 200) {
        throw Exception(result['error'] ?? 'Unknown error occurred');
      }

      return result;
    } catch (e) {
      throw Exception('Failed to run model: $e');
    }
  }

  void dispose() {
    _setupProgressController.close();
    _setupErrorController.close();
    stopService();
  }
}
