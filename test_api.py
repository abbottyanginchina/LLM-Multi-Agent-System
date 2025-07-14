#!/usr/bin/env python3
"""
测试API连接的脚本
"""
import os
import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI

# 加载环境变量
load_dotenv("template.env")  # 明确指定env文件


def test_env_vars():
    """测试环境变量是否正确加载"""
    print("=== 环境变量测试 ===")
    base_url = os.getenv("MINE_BASE_URL")
    api_key = os.getenv("MINE_API_KEYS")  # 注意这里用的是MINE_API_KEYS
    api_key_alt = os.getenv("API_KEY")  # 检查代码中期望的变量名

    print(f"MINE_BASE_URL: {base_url}")
    print(f"MINE_API_KEYS: {api_key[:20] + '...' if api_key else 'None'}")
    print(f"API_KEY: {api_key_alt[:20] + '...' if api_key_alt else 'None'}")

    return base_url, api_key


async def test_api_connection():
    """测试API连接"""
    print("\n=== API连接测试 ===")

    base_url, api_key = test_env_vars()

    if not base_url or not api_key:
        print("❌ 环境变量未正确设置")
        return False

    try:
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
        )

        # 测试简单的聊天完成
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "1+1等于多少？"}],
            max_tokens=50,
        )

        print("✅ API连接成功！")
        print(f"响应: {response.choices[0].message.content}")
        return True

    except Exception as e:
        print(f"❌ API连接失败: {e}")
        return False


async def test_llm_registry():
    """测试LLM注册表"""
    print("\n=== LLM注册表测试 ===")
    try:
        import llm
        from llm.llm_registry import LLMRegistry

        print("可用的LLM:", list(LLMRegistry.registry.keys()))

        # 尝试获取deepseek模型
        if "deepseek" in LLMRegistry.registry:
            deepseek_llm = LLMRegistry.get("deepseek-chat")  # 修改调用方式
            print(f"✅ deepseek LLM创建成功: {deepseek_llm}")

            # 测试异步生成
            try:
                response = await deepseek_llm.agen(
                    [{"role": "user", "content": "2+2等于多少？"}]
                )
                print(f"✅ LLM响应: {response}")
                return True
            except Exception as e:
                print(f"❌ LLM调用失败: {e}")
                return False
        else:
            print("❌ deepseek未在注册表中找到")
            return False

    except Exception as e:
        print(f"❌ LLM注册表测试失败: {e}")
        return False


async def test_agents():
    """测试代理注册表"""
    print("\n=== 代理注册表测试 ===")
    try:
        import agents
        from agents.agent_registry import AgentRegistry

        print("可用的代理:", list(AgentRegistry.registry.keys()))

        # 测试MathSolver
        if "MathSolver" in AgentRegistry.registry:
            math_solver = AgentRegistry.get(
                "MathSolver", domain="gsm8k", llm_name="deepseek"
            )
            print(f"✅ MathSolver创建成功: {math_solver}")

            # 测试执行
            try:
                test_input = {"task": "1+1等于多少？"}
                result = await math_solver.async_execute(test_input)  # 只传递input参数
                print(f"✅ MathSolver响应: {result}")
            except Exception as e:
                print(f"❌ MathSolver执行失败: {e}")

        # 测试FinalRefer
        if "FinalRefer" in AgentRegistry.registry:
            final_refer = AgentRegistry.get(
                "FinalRefer", domain="gsm8k", llm_name="deepseek"
            )
            print(f"✅ FinalRefer创建成功: {final_refer}")

        return True

    except Exception as e:
        print(f"❌ 代理测试失败: {e}")
        return False


async def main():
    """主测试函数"""
    print("开始测试API连接和组件...")

    # 测试API连接
    api_ok = await test_api_connection()

    # 测试LLM注册表
    llm_ok = await test_llm_registry()

    # 测试代理
    agents_ok = await test_agents()

    print(f"\n=== 测试总结 ===")
    print(f"API连接: {'✅' if api_ok else '❌'}")
    print(f"LLM注册表: {'✅' if llm_ok else '❌'}")
    print(f"代理系统: {'✅' if agents_ok else '❌'}")

    if all([api_ok, llm_ok, agents_ok]):
        print("🎉 所有测试通过！")
    else:
        print("⚠️  存在问题需要修复")


if __name__ == "__main__":
    asyncio.run(main())
