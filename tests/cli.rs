use std::process::Command;

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_with-gpu"))
}

#[test]
fn help_is_available_without_nvidia_hardware() {
    let output = binary().arg("--help").output().unwrap();

    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Automatically selects idle GPUs"));
    assert!(stdout.contains("--status"));
}

#[test]
fn version_is_available_without_nvidia_hardware() {
    let output = binary().arg("--version").output().unwrap();

    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).starts_with("with-gpu "));
}

#[test]
fn clap_rejects_invalid_option_relationships() {
    for (args, expected) in [
        (vec!["--json"], "--status"),
        (vec!["--timeout", "1", "true"], "--wait"),
        (vec!["--strict", "true"], "--gpu-type"),
        (vec!["--min-gpus", "0", "true"], "at least 1"),
    ] {
        let output = binary().args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(2));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "stderr did not contain {expected:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn semantic_validation_precedes_gpu_discovery() {
    for (args, expected) in [
        (
            vec!["--min-gpus", "2", "--max-gpus", "1", "true"],
            "min-gpus (2) cannot be greater than max-gpus (1)",
        ),
        (
            vec!["--max-util", "101", "true"],
            "max-util must be between 0 and 100",
        ),
    ] {
        let output = binary().args(args).output().unwrap();
        assert_eq!(output.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(expected),
            "stderr did not contain {expected:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(!String::from_utf8_lossy(&output.stderr).contains("NVML"));
    }
}

#[test]
fn missing_command_is_reported_before_gpu_discovery() {
    let output = binary().output().unwrap();

    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("No command specified"));
    assert!(!stderr.contains("NVML"));
}

#[cfg(target_os = "macos")]
#[test]
fn macos_passthrough_preserves_the_command_exit_code() {
    let status = binary().args(["sh", "-c", "exit 7"]).status().unwrap();

    assert_eq!(status.code(), Some(7));
}

#[cfg(target_os = "macos")]
#[test]
fn macos_passthrough_reports_a_missing_executable() {
    let output = binary()
        .arg("with-gpu-command-that-does-not-exist")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&output.stderr).contains("Failed to execute command"));
}
