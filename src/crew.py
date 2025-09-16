# src/crew.py
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from .tools.tools import data_ingestion_tool, liquidity_analysis_tool, report_generation_tool, chart_generation_tool

# Note: keep the model consistent with your account; the agents call local tools for heavy lifting.
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

data_ingestor = Agent(
    role="Data Ingestion Specialist",
    goal="Extract data from files into structured tables.",
    backstory="Expert in financial data extraction and cleaning.",
    tools=[data_ingestion_tool],
    llm=llm,
    verbose=True,
    max_iter=1,
    max_rpm=1
)

analyst = Agent(
    role="Liquidity Risk Analyst",
    goal="Assess liquidity risk from ingested data and produce structured analysis.",
    backstory="Specialist in treasury risk and financial modelling.",
    tools=[liquidity_analysis_tool],
    llm=llm,
    verbose=True,
    max_iter=1,
    max_rpm=1
)

reporter = Agent(
    role="Report Generator",
    goal="Create an adaptive markdown report and render charts.",
    backstory="Professional report builder who embeds charts and tables.",
    tools=[report_generation_tool, chart_generation_tool],
    llm=llm,
    verbose=True,
    max_iter=1,
    max_rpm=1
)

def create_crew(file_paths: list, objective: str):
    """
    Orchestrates: ingest -> analyze -> report.
    Returns final markdown report string.
    """
    agents_list = [data_ingestor, analyst, reporter]

    ingestion_task = Task(
        description=f"Ingest data from file paths: {file_paths}",
        expected_output="JSON dict of all sheets",
        agent=data_ingestor,
        return_direct=True
    )

    analysis_task = Task(
        description=f"Analyze ingested JSON data for objective: {objective}",
        expected_output="Analysis dict with metrics, tables, chart_specs, and insights",
        agent=analyst,
        context=[ingestion_task]
    )

    report_task = Task(
        description=f"Generate final report markdown using analysis and generate charts",
        expected_output="Markdown report string",
        agent=reporter,
        context=[analysis_task]
    )

    crew = Crew(
        agents=agents_list,
        tasks=[ingestion_task, analysis_task, report_task],
        process=Process.sequential,
        verbose=True
    )

    # Kick off: pass file_paths and objective into the kickoff inputs
    return crew.kickoff(inputs={"file_paths": file_paths, "objective": objective})
