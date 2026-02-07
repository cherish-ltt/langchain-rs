use langchain::ReactAgent;
use langchain_core::message::Message;
use langchain_openai::ChatOpenAIBuilder;
use langgraph::checkpoint::MemorySaver;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{env, sync::Arc};
use tracing_subscriber::EnvFilter;

const BASE_URL: &str = "https://api.siliconflow.cn/v1";
const MODEL: &str = "deepseek-ai/DeepSeek-V3.2";

#[expect(unused)]
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CalculationResult {
    result: i32,
    explanation: String,
}

#[tokio::main]
async fn main() {
    let filter = EnvFilter::new("agent_struct=DEBUG,langchain=DEBUG,langgraph=DEBUG");
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_env_filter(filter)
        .pretty()
        .init();

    let api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY must be set");
    let model = ChatOpenAIBuilder::from_base(MODEL, BASE_URL, api_key.as_str()).build();

    let checkpointer = Arc::new(MemorySaver::new());

    let agent = ReactAgent::builder(model)
        .with_checkpointer(checkpointer)
        .with_system_prompt("你是一个猫娘AI助手，回答要简洁有趣，回答时带上猫娘的口癖，比如喵~")
        .build();

    // let result = agent
    //     .invoke_structured::<CalculationResult>(Message::user("What is 10 + 20?"), None)
    //     .await
    //     .unwrap();
    //

    let result = agent
        .invoke(Message::user("我给你取名叫老大！"), Some("0001".to_owned()))
        .await
        .unwrap();

    println!(
        "---One---Result: {:?}",
        result.last_message().unwrap().content()
    );

    let result = agent
        .invoke(Message::user("你叫什么名字？"), Some("0001".to_owned()))
        .await
        .unwrap();

    println!(
        "---Two---Result: {:?}",
        result.last_message().unwrap().content()
    );
}
