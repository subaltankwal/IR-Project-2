import pysolr

# Create a connection to Solr
solr = pysolr.Solr('http://localhost:8983/solr/localDocs', always_commit=True)

# Perform a search on a particular field
results = solr.search('Title:how')

# Iterate over the results
for result in results:
    print(result['id'])
