import pprint
from time import sleep
from dotenv import load_dotenv

from just_agents import llm_options
from just_agents.base_agent import ChatAgent, ChatAgentWithLogging
from just_agents.data_classes import ImageContent, Message, Role, TextContent
from just_agents.llm_options import LLAMA3_3, LLAMA4_SCOUT, LLAMA4_MAVERICK, OPENAI_GPT5_NANO, GEMINI_2_5_FLASH
from pprint import pprint

from just_agents.tools.db import sqlite_query
import pytest
import requests
import os
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional


db_path = Path("tests/data/open_genes.sqlite")
sleep_time = float(os.getenv("SLEEP_TIME", "20.0"))



@pytest.fixture(scope="session")
def open_genes_db() -> Path:
    """Downloads the open_genes.sqlite database if not present and returns the path"""
    # Define the database path
    db_path = Path("tests/data/open_genes.sqlite").absolute().resolve()

    
    # Create directories if they don't exist
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Download file if it doesn't exist
    if not db_path.exists():
        url = "https://github.com/longevity-genie/longevity_gpts/raw/main/open_genes/data/open_genes.sqlite"
        print(f"Downloading database from {url}")
        
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # Save the file
        with open(db_path, "wb") as f:
            f.write(response.content)
        
        print(f"Database saved to {db_path}")
    
    return db_path


def _test_database_tables(agent: ChatAgent, open_genes_db: Path):
    """Helper function to test database table queries with different agents"""
    response = agent.query("Show me all tables in the database")
    agent.memory.pretty_print_all_messages()
    assert response is not None, "Response should not be None"
    
    # Add specific table checks
    expected_tables = [
        'lifespan_change',
        'gene_criteria',
        'gene_hallmarks',
        'longevity_associations'
    ]
    
    # Check each expected table is in the agent's response
    for table in expected_tables:
        assert table.lower() in response.lower(), f"Expected table '{table}' not found in agent response"
    
    sleep(sleep_time)

def test_quering(open_genes_db):
    load_dotenv(override = True)
    agent = ChatAgent(role="helpful agent which knows how operate with databases",
                    goal=f"help users by using SQL syntax to form commands to work with the {open_genes_db} sqlite database",
                    task="formulate appropriate commands to operate in the given database and always include the table names in your response.",
                    tools=[sqlite_query],
                    llm_options=LLAMA4_SCOUT,
                    key_list_env="GROQ_API_KEY"
                    )
    _test_database_tables(agent, open_genes_db)

def test_querying_gemini(open_genes_db):
    load_dotenv(override = True)
    agent = ChatAgentWithLogging(role="helpful agent which knows how operate with databases",
                    goal=f"help users by using SQL syntax to form commands to work with the {open_genes_db} sqlite database",
                    task="formulate appropriate commands to operate in the given database using 'sqlite_query' tool and always include the table names in your response.",
                    tools=[sqlite_query],
                    llm_options=GEMINI_2_5_FLASH
                    )
    _test_database_tables(agent, open_genes_db)


def test_query_structural():
    load_dotenv(override=True)
    agent = ChatAgentWithLogging(role="helpful agent that provides structured information",
                                 goal="help users by providing structured information in JSON format",
                                 task="analyze user questions and provide comprehensive responses in a structured format",
                                 format="""all your answers should be represented as a JSON object with the following structure:
                    {
                      "user_question": "the original question asked by the user",
                      "answer": "your analysis of the question",
                      "delegate_to": "",
                      "question": "",
                      "final_answer": "your complete answer to the user's question"
                    }""",
                                 #llm_options=LLAMA3_3
                                 #llm_options=llm_options.OPENAI_GPT4o,
                                 llm_options=GEMINI_2_5_FLASH
                                 )
    
    # Ask the agent a question that doesn't require SQL or tool use
    response = agent.query_structural("What are the main factors that contribute to aging?",
                                      parser=AgentResponse, enforce_validation=True)
    
    print("RESPONSE 1 =======================================")
    pprint(response)

    # Validate response structure
    assert isinstance(response, AgentResponse), "Response should be an AgentResponse instance"
    assert response.user_question == "What are the main factors that contribute to aging?"
    # Check that the response follows the expected structure
    assert hasattr(response, "user_question")
    assert hasattr(response, "answer")
    assert hasattr(response, "delegate_to")
    assert hasattr(response, "question")
    #assert hasattr(response, "final_answer")


class Annotation(BaseModel):
    abstract: str = Field(...)
    authors: List[str] = Field(default_factory=list)
    title: str = Field(...)
    source: str = Field(...)

#@pytest.mark.skip(reason="until fixed in https://github.com/BerriAI/litellm/issues/7808")
def test_gemini_summarization():
    load_dotenv(override=True)
    agent = ChatAgentWithLogging(
        llm_options=llm_options.GEMINI_2_5_FLASH,
        #llm_options=llm_options.OPENAI_GPT4o,
        tools=[],
        system_prompt="""You are a paper annotator. You extract the abstract, authors and titles of the papers.
        Abstract and authors must be exactly he way they are in the paper, do not edit them.
        You provide your output as json object of the following JSON format:
        {
            "abstract": "...",
            "authors": ["...", "..."],
            "title": "...",
            "source": "...",
        }
        Make sure to provide the output in the correct format, do not add any other text or comments, do not add ```json or other surrounding.
        Make sure you handle the multiline strings properly, according to the JSON format.
        For string either use one line or use proper escape characters (\n) for line breaks to not break the JSON format.
        Make sure to provide the output in the correct format, do not add any other text or comments.
        For source you either give DOI, pubmed or filename (if doi or pubmed is not available).
        File filename you give a filename of the file in the folder together with the extension.""",
        )

    query = """Extract the abstract, authors and title of the following paper (from file 2023-Situational Awareness and Proactive Engagement Predict Higher Time in Range in Adolescents and Young Adults Using Hybrid Closed-Loop.md):\n# Situational Awareness and Proactive Engagement Predict Higher Time in Range in Adolescents and Young Adults Using Hybrid Closed-Loop  \n\nLaurel H. Messer1,2, Paul F. Cook2, Stephen Voida3, Casey Fiesler3, Emily Fivekiller1, Chinmay Agrawal4, Tian $\\pmb{\\chi}_{\\mathbf{u}^{3}}$ , Gregory P. Forlenza1, Sriram Sankaranarayanan4  \n\n1Barbara Davis Center for Diabetes, University of Colorado Anschutz Medical Campus, Aurora, CO, USA  \n\n2College of Nursing, University of Colorado Anschutz Medical Campus, Aurora, CO, USA  \n\n3Department of Information Science, University of Colorado Boulder, Boulder, CO, USA  \n\n4Department of Computer Science, University of Colorado Boulder, Boulder, CO, USA  \n\n# Abstract  \n\nBackground.—Adolescents and young adults with type 1 diabetes have high HbA1c levels and often struggle with self-management behaviors and attention to diabetes care. Hybrid closed-loop systems (HCL) like the t:slim X2 with Control-IQ technology (Control-IQ) can help improve glycemic control. The purpose of this study is to assess adolescents’ situational awareness of their glucose control and engagement with the Control-IQ system to determine significant factors in daily glycemic control.  \n\nMethods.—Adolescents (15–25 years) using Control-IQ participated in a 2-week prospective study, gathering detailed information about Control-IQ system engagements (boluses, alerts, and so on) and asking the participants’ age and gender about their awareness of glucose levels 2–3 times/day without checking. Mixed models assessed which behaviors and awareness items correlated with time in range (TIR, 70–180 mg/dl, 3.9–10.0 mmol/L).  \n\nResults.—Eighteen adolescents/young adults (mean age $18\\pm1.86$ years and $86\\%$ White nonHispanic) completed the study. Situational awareness of glucose levels did not correlate with time since the last glucose check $\\zeta\\!=\\!0.8)$ ). In multivariable modeling, lower TIR was predicted on days when adolescents underestimated their glucose levels $\\stackrel{r}{}=-0.22)$ , received more CGM alerts $(r\\!=-0.31)$ ), and had more pump engagements $(r\\!=-\\!0.27)$ . A higher TIR was predicted when adolescents responded to CGM alerts $r\\!=\\!0.20)$ and entered carbohydrates into the bolus calculator $(r\\!=0.49)$ ).  \n\nConclusion.—Situational awareness is an independent predictor of TIR and may provide insight into patterns of attention and focus that could positively influence glycemic outcomes in adolescents. Proactive engagements predict better TIR, whereas reactive engagement predicted lower TIR. Future interventions could be designed to train users to develop awareness and expertise in effective diabetes self-management.  \n\n# 1.  Introduction  \n\nAdolescents and emergent adults (ages 15–25 years) with T1D have the highest average HbA1c levels of any age group with diabetes, peaking at $9.3\\%$ , well above goal of $7.0\\%$ for most people with diabetes [1]. Diabetes technologies such as hybrid closed-loop systems (HCL) can improve glycemic control in children, adolescents, and adults with diabetes [2–5]. HCL systems partially automate insulin delivery with algorithms that use sensor glucose input to administer insulin doses aimed at keeping glucose levels in target range. The Tandem t:slim X2 with Control-IQ technology (referred to here as “Control-IQ”) is one of these HCL systems. The Control-IQ system consists of an insulin pump that implements the Control-IQ algorithm combined with a Dexcom G6 continuous glucose monitor (CGM) [6]. Persons with diabetes who use the Control-IQ system wear the system continuously and direct the pump to deliver insulin boluses for meals and hyperglycemia as needed.  \n\nWhile the Control-IQ system improves glycemic control in adolescents, user behavior and engagement remain important to achieving optimal glycemia [7, 8]. Diabetes selfmanagement behaviors are particularly difficult for adolescents and young adults due to a variety of developmental, cognitive, and psychological factors unique to the age group [9–11]. Engagement with HCL systems like Control-IQ (e.g. giving insulin boluses, monitoring glucose levels, and so on) is one subset of self-management, together with other behaviors like food selection or timing, and physical activity. We have previously shown how adolescents and young adults have high interpersonal and intrapersonal variability in their diabetes self-management behaviors, and how a variety of biopsychosocial daily factors correlate with these fluctuations [12]. Therefore, more research about their engagement with their diabetes care is warranted.  \n\nBecause adolescents face many competing challenges for attention, we examined how “situationally aware” adolescents and young adults were in relation to their diabetes care throughout the day. Situational awareness is defined as a combination of (a) knowing numerous pieces of data, (b) having a deep understanding of context, and (c) being able to project future states in reference to present goals [13]. Situational awareness is associated with expertise, and the related cognitive processes often bypass conscious awareness  \n\n[14]. In the context of diabetes, situational awareness refers to “strategic” awareness of current health states, an understanding of what they mean, and the ability to execute self-management behaviors that affect them. Among adults with diabetes, greater skill in recognizing glucose problems was linked to better glucose control in a way that declarative knowledge about diabetes was not [15]. Situational awareness involves automatic perception and attention processes that we have characterized as belonging to the “Intuitive mind,” which can be differentiated from factual knowledge and intentions at the more conscious “narrative mind” level [16].  \n\nThe purpose of this study, therefore, was to assess adolescents’ engagement with the Control-IQ system, and their situational awareness of glucose levels throughout the day, and to evaluate these variables’ effects on glycemic control. Identifying patterns of awareness, engagement, and glycemia is a first step to understand how adolescents and young adults can more effectively manage their diabetes using the Control-IQ system.  \n\n# 2.  Methods  \n\nWe conducted a prospective, 2-week study involving adolescents and young adults recruited from the Barbara Davis Center to collect data related to diabetes engagement, situational awareness, and glucose control. The Colorado Multiple Institutional Review Board approved this research. Participants were ages 15–25 years old inclusive, had a diagnosis of type 1 diabetes, and used the Control-IQ system to manage their diabetes. Our intention was to recruit individuals with diverse HbA1c levels at baseline, so potential participants were prescreened for this and use of Control-IQ. We chose Control-IQ as the HCL system of interest because it was the most widely used HCL in our clinic at the time of the study. Additionally, participants had to be using a commercial iOS (Apple) iPhone device with the HealthKit application and be willing to wear a compatible smartwatch for the duration of the study.  \n\n# 2.1.  Procedures.  \n\nParticipants were enrolled in this study for 2-weeks and wore their Control-IQ system continuously. Although during routine use, Control-IQ users can check glucose levels on the insulin pump itself or a separate CGM app on a phone, we asked participants to only check glucose levels on their insulin pump so we could collect data about these interactions from the pump itself to better quantify user engagement. Throughout the study period, participants were sent 2–3 quasi-randomly timed “situational awareness” surveys that asked about their current awareness of glucose levels and predictions for future glucose levels.  \n\n# 2.2.  Data Collection  \n\n2.2.1.  Situational Awareness Questionnaire.—Situational awareness can be difficult to measure, but studies have shown that “awareness-in-the-moment” measures are more correlated with performance than a participant’s subjective rating of how situationally aware they were after the fact [17]. We therefore assessed situational awareness with a 4-item survey delivered at random times and asked participants: (1) when did you last view your CGM glucose value? (In the past 15 minutes, past hour, past 2 hours, past 3 hours, and longer than 3 hours); (2) without looking, what is your glucose now? ${\\leq}70\\;\\mathrm{mg/dl}$ , 71–120 mg/dl, 121–180 mg/dl, $181{-}250\\;\\mathrm{mg/dl}$ , and ${>}250\\ \\mathrm{mg/dl}$ ); and (3) without looking, what direction is it trending? (going up, going down, and staying stable); and (4) after looking at your CGM, what do you think your glucose level will be 1 hour from now? $(<\\!70\\;\\mathrm{mg/dl}$ , $71{-}120\\;\\mathrm{mg/dl}$ , $121{-}180\\;\\mathrm{mg/dl}$ , $181{-}250\\;\\mathrm{mg/dl}$ , and $>\\!250\\;\\mathrm{mg/dl}$ ). Items 1 and 2 correspond to knowledge of data, item 3 reflects contextual knowledge, and item 4 requires prediction, which are the three major components of situational awareness. We expected that all of these components would predict glycemic control.  \n\n2.2.2.  Engagement with CGM.—Engagement behaviors were quantified from ControlIQ downloads, including the daily number of interactions with the Control-IQ system (including checking glucose levels, responding to alerts, giving meal boluses, and pump maintenance), the number of CGM alerts, the percent of CGM alerts acknowledged by the user, the number of boluses given, the number of grams of carbohydrate entered into the pump each day, and the number and percent of boluses that included a carbohydrate entry. Each of these system device functions can be part of a participant’s diabetes selfmanagement approach, but at the outset we did not have any clear expectation about which of them might be most strongly correlated with glycemic control.  \n\n2.2.3.  Glucose Outcome.—Daily glucose data were collected from the CGM, with time-in-range (TIR, $70{-}180\\;\\math'}"""
    
    enforce_validation = True
    
    response = agent.query_structural(
                query,
                Annotation,
                enforce_validation=enforce_validation)
    
    validated_response = Annotation.model_validate(response)
    assert isinstance(validated_response, Annotation), "Response should be an Annotation instance"
    assert "Situational Awareness" in validated_response.title, "it must have Situational Awareness in the title"
    


class AgentResponse(BaseModel):
    user_question: str = Field(..., description="The original question asked by the user")
    answer: Optional[str] = Field(default="", description="Agent's initial answer or analysis")
    delegate_to: Optional[str] = Field(default="", description="Name of the agent to delegate to, empty if no delegation")
    question: Optional[str] = Field(default="", description="Question to delegate to another agent, empty if no delegation")
    final_answer: Optional[str] = Field(default="", description="Final answer to the user's question, empty if pending delegation")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_question": "What genes extend lifespan?",
                "answer": "This requires database access to analyze lifespan data",
                "delegate_to": "agent_db",
                "question": "SELECT * FROM lifespan_change WHERE species='mouse' ORDER BY lifespan_change DESC",
                "final_answer": ""
            }
        }
    )


def test_vision():
    load_dotenv(override = True)
    agent = ChatAgentWithLogging(role="helpful agent that can see",
                goal="help users by providing a description of the image",
                task="analyze the image and provide a description of the image",
                llm_options=GEMINI_2_5_FLASH
                )
    message = Message(
        role=Role.user,
        content=[
        TextContent(text="What is in this image?"),
        ImageContent(image_url="https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg", use_nested_format=True)
        ]
    )

    result = agent.query(message,remember_query=False)
    #https://arxiv.org/pdf/2410.05780
    assert "cat" in result or "kitten" in result
    assert "tabby" in result or "stripe" in result
    result = agent.query(message.model_dump_json(),remember_query=False)
    assert "cat" in result or "kitten" in result
    assert "tabby" in result or "stripe" in result



@pytest.mark.skip(reason="uses lots of tokens")
def test_delegation():
    load_dotenv(override=True)
    agent_db = ChatAgent(role="helpful agent which knows how operate with databases",
                    goal=f"help users by using SQL syntax to form commands to work with the {db_path} sqlite database",
                    task="formulate appropriate commands to operate in the given database.",
                    tools=[sqlite_query],
                    llm_options=GEMINI_2_5_FLASH
                    )
    agent_db.query("What is in this image?")

    ponder_agent = ChatAgent(
        role="helpful agent which will only distribute the tasks to it's calling list and make minimal suggestions",
        goal="help users by guiding them to appropriate instances that will answer their questions",
        task=f"You are generalist agent which can delegate tasks to other agents, so far you only know agent_db which can run database queries on your behalf.",
        format="""all your answers you represent solely as JSON object without any text either before or after it. The format is the following: 
        {
        "user_question": "question",
        "answer": "your answer", 
        "delegate_to": "agent_name", 
        "question": "question_to_delegate",
        "final_answer": "final answer to the question"
        } If you do not delegate then delegate_to and question should be empty. if you are ready to answer the question then final_answer should be your answer, if not it must be empty""",
        llm_options=OPENAI_GPT5_NANO
    )

    for i in range(2):
        result = ponder_agent.query_structural(
            "Interventions on which genes extended mice lifespan most of all? Search all the relevant tables in the open-genes sqlite and only for mouse", 
            parser=AgentResponse, 
            enforce_validation=False
        )
        if result.delegate_to == "agent_db":
            result = agent_db.query(result.question)
        print("\nQuery Result:")  # Add a header for clarity
        sleep(sleep_time)
        pprint(result)
        