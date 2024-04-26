import pysolr

solr = pysolr.Solr('http://localhost:8983/solr/localDocs', timeout=10)
solr.delete(q='*:*')
solr.commit()
