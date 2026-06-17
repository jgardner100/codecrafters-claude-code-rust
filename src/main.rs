use async_openai::{Client, config::OpenAIConfig};
use clap::Parser;
use serde_json::{Value, json};
use std::{env, process, fs};

#[derive(Parser)]
#[command(author, version, about)]
struct Args {
    #[arg(short = 'p', long)]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let base_url = env::var("OPENROUTER_BASE_URL")
        .unwrap_or_else(|_| "https://openrouter.ai/api/v1".to_string());

    let api_key = env::var("OPENROUTER_API_KEY").unwrap_or_else(|_| {
        eprintln!("OPENROUTER_API_KEY is not set");
        process::exit(1);
    });

    let config = OpenAIConfig::new()
        .with_api_base(base_url)
        .with_api_key(api_key);

    let client = Client::with_config(config);

    // Initialize the conversation with the user prompt
    let mut messages = vec![
        json!({
            "role": "user",
            "content": args.prompt
        })
    ];

    eprintln!("Starting agent loop with prompt: {}", args.prompt);

    // Agent loop
    loop {
        eprintln!("Sending request to API with {} messages", messages.len());

        // Call the API with current messages
        let response: Value = client
            .chat()
            .create_byot(json!({
                "messages": messages,
                "model": "anthropic/claude-haiku-4.5",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "Read",
                            "description": "Read and return the contents of a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path to the file to read"
                                    }
                                },
                                "required": ["file_path"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Write",
                            "description": "Write content to a file",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "The path of the file to write to"
                                    },
                                    "content": {
                                        "type": "string",
                                        "description": "The content to write to the file"
                                    }
                                },
                                "required": ["file_path", "content"]
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "Bash",
                            "description": "Execute a shell command",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "command": {
                                        "type": "string",
                                        "description": "The command to execute"
                                    }
                                },
                                "required": ["command"]
                            }
                        }
                    }
                ]
            }))
            .await?;

        // Extract the assistant's response message
        let assistant_message = response["choices"][0]["message"].clone();
        
        // Add the assistant's response to messages
        messages.push(assistant_message.clone());

        eprintln!("Received response from API");

        // Check the finish_reason
        let finish_reason = response["choices"][0]["finish_reason"].as_str().unwrap_or("");
        
        // Check if there are tool calls in the response
        if let Some(tool_calls) = assistant_message["tool_calls"].as_array() {
            if !tool_calls.is_empty() {
                eprintln!("Found {} tool calls", tool_calls.len());

                // Process each tool call
                for tool_call in tool_calls {
                    let tool_call_id = tool_call["id"].as_str().unwrap_or("unknown");
                    let function_name = tool_call["function"]["name"].as_str().unwrap_or("");
                    let arguments_str = tool_call["function"]["arguments"].as_str().unwrap_or("{}");
                    
                    eprintln!("Executing tool: {} with id: {}", function_name, tool_call_id);

                    // Parse the arguments
                    let arguments = serde_json::from_str::<Value>(arguments_str).unwrap_or(json!({}));

                    // Execute the appropriate tool
                    let tool_result = match function_name {
                        "Read" => {
                            if let Some(file_path) = arguments["file_path"].as_str() {
                                eprintln!("Reading file: {}", file_path);
                                match fs::read_to_string(file_path) {
                                    Ok(contents) => contents,
                                    Err(e) => format!("Error reading file: {}", e),
                                }
                            } else {
                                "Error: file_path not provided".to_string()
                            }
                        },
                        "Write" => {
                            if let (Some(file_path), Some(content)) = 
                                (arguments["file_path"].as_str(), arguments["content"].as_str()) {
                                eprintln!("Writing file: {}", file_path);
                                match fs::write(file_path, content) {
                                    Ok(_) => "File written successfully".to_string(),
                                    Err(e) => format!("Error writing file: {}", e),
                                }
                            } else {
                                "Error: file_path or content not provided".to_string()
                            }
                        },
                        "Bash" => {
                            if let Some(command) = arguments["command"].as_str() {
                                eprintln!("Executing command: {}", command);
                                match std::process::Command::new("bash")
                                    .arg("-c")
                                    .arg(command)
                                    .output() {
                                    Ok(output) => {
                                        let stdout = String::from_utf8_lossy(&output.stdout);
                                        let stderr = String::from_utf8_lossy(&output.stderr);
                                        if !stderr.is_empty() {
                                            format!("{}\n{}", stdout, stderr)
                                        } else {
                                            stdout.to_string()
                                        }
                                    },
                                    Err(e) => format!("Error executing command: {}", e),
                                }
                            } else {
                                "Error: command not provided".to_string()
                            }
                        },
                        _ => format!("Unknown tool: {}", function_name),
                    };

                    // Add tool result to messages
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": tool_result
                    }));

                    eprintln!("Tool execution completed for: {}", tool_call_id);
                }
            } else {
                // No tool calls, print the content and exit
                if let Some(content) = assistant_message["content"].as_str() {
                    println!("{}", content);
                }
                break;
            }
        } else {
            // No tool calls field, check if we have content to print
            if let Some(content) = assistant_message["content"].as_str() {
                if !content.is_empty() {
                    println!("{}", content);
                }
            }
            // Check finish_reason to decide if we should continue
            if finish_reason == "stop" || finish_reason == "end_turn" {
                break;
            }
        }

        // Break if finish_reason indicates completion
        if finish_reason == "stop" {
            break;
        }
    }

    Ok(())
}
