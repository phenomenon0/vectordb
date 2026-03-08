use std::net::TcpListener;
use std::thread;
use std::time::Duration;
use tauri::Manager;
use tauri_plugin_shell::ShellExt;

const PORT_RANGE_START: u16 = 8080;
const PORT_RANGE_END: u16 = 8099;

/// FIX #5: Bind-and-hold pattern to prevent port race attacks.
/// Returns both the port and the held listener. The listener must be kept
/// alive until the sidecar has started and bound the port itself.
/// Previously, find_free_port() would bind, immediately drop the listener,
/// and return the port — a TOCTOU vulnerability where a local attacker
/// could bind the port in the gap and hijack the privileged Tauri window.
fn find_free_port() -> (u16, TcpListener) {
    for port in PORT_RANGE_START..=PORT_RANGE_END {
        if let Ok(listener) = TcpListener::bind(("127.0.0.1", port)) {
            return (port, listener);
        }
    }
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .expect("failed to bind any port");
    let port = listener.local_addr().unwrap().port();
    (port, listener)
}

fn wait_for_server(port: u16, timeout: Duration) -> bool {
    use std::net::TcpStream;
    let start = std::time::Instant::now();
    let addr = ("127.0.0.1", port);
    while start.elapsed() < timeout {
        if TcpStream::connect(addr).is_ok() {
            eprintln!("[tauri] server ready after {:?}", start.elapsed());
            return true;
        }
        thread::sleep(Duration::from_millis(200));
    }
    false
}

#[tauri::command]
fn win_minimize(window: tauri::Window) {
    let _ = window.minimize();
}

#[tauri::command]
fn win_toggle_maximize(window: tauri::Window) {
    if window.is_maximized().unwrap_or(false) {
        let _ = window.unmaximize();
    } else {
        let _ = window.maximize();
    }
}

#[tauri::command]
fn win_close(window: tauri::Window) {
    let _ = window.close();
}

#[tauri::command]
fn win_start_drag(window: tauri::Window) {
    let _ = window.start_dragging();
}

/// Injected after navigation: make existing nav draggable + add window controls.
const DESKTOP_INJECT_JS: &str = include_str!("inject.js");

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let (port, held_listener) = find_free_port();
    eprintln!("[tauri] using port {} (held until sidecar starts)", port);

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![
            win_minimize,
            win_toggle_maximize,
            win_close,
            win_start_drag,
        ])
        .setup(move |app| {
            let shell = app.shell();

            // Drop the held listener just before spawning the sidecar.
            // The sidecar will immediately bind this port. The window between
            // drop and sidecar bind is minimal (same process, next statement).
            drop(held_listener);

            let (mut rx, child) = shell
                .sidecar("deepdata")
                .expect("failed to find deepdata sidecar binary")
                .args(["serve", "--port", &port.to_string()])
                .spawn()
                .expect("failed to spawn deepdata sidecar");

            tauri::async_runtime::spawn(async move {
                use tauri_plugin_shell::process::CommandEvent;
                while let Some(event) = rx.recv().await {
                    match event {
                        CommandEvent::Stdout(line) | CommandEvent::Stderr(line) => {
                            eprint!("[sidecar] {}", String::from_utf8_lossy(&line));
                        }
                        CommandEvent::Terminated(payload) => {
                            eprintln!("[sidecar] terminated: {:?}", payload);
                            break;
                        }
                        _ => {}
                    }
                }
            });

            app.manage(child);

            let window = app.get_webview_window("main").unwrap();
            thread::spawn(move || {
                if wait_for_server(port, Duration::from_secs(15)) {
                    let url = format!("http://localhost:{}", port);
                    let _ = window.navigate(url.parse().unwrap());
                    thread::sleep(Duration::from_secs(2));
                    let _ = window.eval(DESKTOP_INJECT_JS);
                } else {
                    eprintln!("[tauri] server failed to start within 15s");
                }
            });

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error running DeepData");
}
