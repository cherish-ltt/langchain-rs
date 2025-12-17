use langchain::{ReactAgent, Tool, tool, tools_from_fns};
use langchain_core::{message::Message, state::MessageState};
use langchain_openai::ChatOpenAiModel;
use langgraph::node::NodeRunError;

const BASE_URL: &str = "https://api.siliconflow.cn/v1";
const API_KEY: &str = "";
const MODEL: &str = "deepseek-ai/DeepSeek-V3.2";

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = ChatOpenAiModel::new(BASE_URL.to_string(), API_KEY.to_string(), MODEL.to_string());
    let mut agent =
        ReactAgent::create_agent_with_tools(model, tools_from_fns!(get_weather, add, sub));

    let result = agent
        .invoke(Message::user(
            "查询成都天气情况,并且计算100+200,并且计算100-200",
        ))
        .await?;

    let message = result
        .messages
        .last()
        .ok_or(anyhow::anyhow!("No message found"))?;
    println!("运行结束{:?}", message);

    Ok(())
}

#[tool(
    description = "获取指定城市的天气信息",
    args(location = "城市名称，例如 '成都'")
)]
async fn get_weather(location: String) -> Result<String, NodeRunError> {
    Ok(format!("Weather in {} is sunny", location))
}

#[tool(description = "计算两个数的和", args(a = "第一个数", b = "第二个数"))]
async fn add(a: f64, b: f64) -> Result<f64, NodeRunError> {
    Ok(a + b)
}

#[tool(description = "计算两个数的差", args(a = "第一个数", b = "第二个数"))]
async fn sub(state: &MessageState, a: f64, b: f64) -> Result<f64, NodeRunError> {
    println!("调用了几次LLM: {:?}", state.llm_calls);
    Ok(a - b)
}
