FROM rayproject/ray:2.41.0

ENV PIP_ROOT_USER_ACTION=ignore
COPY requirements.txt .
RUN pip install -r requirements.txt