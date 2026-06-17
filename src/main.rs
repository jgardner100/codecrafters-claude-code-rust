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

    #[allow(unused_variables)]
    let response: Value = client
        .chat()
        .create_byot(json!({
            "messages": [
                {
                    "role": "user",
                    "content": args.prompt
                }
            ],
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
                }
            ]
        }))
        .await?;

    // You can use print statements as follows for debugging, they'll be visible when running tests.
    eprintln!("Logs from your program will appear here!");

    // Check if there are tool calls in the response
    if let Some(tool_calls) = response["choices"][0]["message"]["tool_calls"].as_array() {
        if !tool_calls.is_empty() {
            // Extract the first tool call
            let tool_call = &tool_calls[0];
            
            // Get the function name
            if let Some(function_name) = tool_call["function"]["name"].as_str() {
                // Get the arguments as a JSON string
                if let Some(arguments_str) = tool_call["function"]["arguments"].as_str() {
                    // Parse the arguments JSON
                    if let Ok(arguments) = serde_json::from_str::<Value>(arguments_str) {
                        // Handle the Read tool
                        if function_name == "Read" {
                            if let Some(file_path) = arguments["file_path"].as_str() {
                                // Read the file and print its contents
                                match fs::read_to_string(file_path) {
                                    Ok(contents) => println!("{}", contents),
                                    Err(e) => {
                                        eprintln!("Error reading file: {}", e);
                                        process::exit(1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else if let Some(content) = response["choices"][0]["message"]["content"].as_str() {
        // If there are no tool calls, print the message content
        println!("{}", content);
    }

    Ok(())
}
