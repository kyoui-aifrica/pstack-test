FROM opensearchproject/opensearch:3.9.0
RUN bin/opensearch-plugin install --batch analysis-nori
