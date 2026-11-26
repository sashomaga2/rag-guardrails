import pytest

from northwind_rag.config import Settings
from northwind_rag.graph import build_graph
from northwind_rag.pipeline import Pipeline


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    return Settings(offline=True, chroma_dir=tmp_path_factory.mktemp("chroma"))


@pytest.fixture(scope="session")
def pipeline(settings) -> Pipeline:
    return Pipeline.create(settings)


@pytest.fixture
def pipeline_with_chat(pipeline):
    """Same KB and guards, but a scripted generator model."""

    def make(chat_model) -> Pipeline:
        suite = type(pipeline.suite)(**{**pipeline.suite.__dict__, "chat": chat_model})
        return Pipeline(
            pipeline.settings, suite, pipeline.kb,
            build_graph(suite, pipeline.kb, pipeline.settings, pipeline.live_agent), pipeline.live_agent,
        )

    return make
