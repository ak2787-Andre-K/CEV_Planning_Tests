//! Prints every Zenoh message it hears, with its key, so you can see what other nodes publish:
//! `listen [<key expression>] [--full]`. The key expression defaults to `**` (everything); payloads are shown as text
//! if they are text (cut to 300 characters unless `--full`), otherwise as their size and first bytes in hex.
use std::time::Instant;

use cev_sst::zenoh_config;
use zenoh::Wait;

fn main() -> zenoh::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let full = args.iter().any(|a| a == "--full");
    // The first argument that isn't a flag (or a --connect / --listen endpoint) is the key expression
    let mut key = "**".to_string();
    let mut skip = false;
    for a in &args {
        if skip {
            skip = false;
        } else if a == "--connect" || a == "--listen" {
            skip = true;
        } else if !a.starts_with("--") {
            key = a.clone();
            break;
        }
    }
    let session = zenoh::open(zenoh_config()?).wait()?;
    let subscriber = session.declare_subscriber(&key).wait()?;
    println!("Listening on '{key}' (Ctrl+C to stop)...");
    let start = Instant::now();
    while let Ok(sample) = subscriber.recv() {
        let bytes = sample.payload().to_bytes();
        let shown = match std::str::from_utf8(&bytes) {
            Ok(text) if full || text.len() <= 300 => text.to_string(),
            Ok(text) => format!("{}... ({} bytes)", &text[..text.floor_char_boundary(300)], text.len()),
            Err(_) => {
                let head: Vec<String> = bytes.iter().take(24).map(|b| format!("{b:02x}")).collect();
                format!("<{} bytes of binary: {}{}>", bytes.len(), head.join(" "), if bytes.len() > 24 { " ..." } else { "" })
            }
        };
        println!("[{:7.2} s] {}: {shown}", start.elapsed().as_secs_f64(), sample.key_expr());
    }
    Ok(())
}
