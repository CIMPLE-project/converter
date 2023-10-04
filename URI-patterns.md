URI patterns for CIMPLE data
==============================

This is the documentation of the URI design pattern used by the CIMPLE Knowledge Graph. The following SPARQL query provides the number of instances of each type ([results](https://data.cimple.eu/sparql?default-graph-uri=&query=SELECT+DISTINCT+count%28%3Fs%29+as+%3Fnb+%3Ftype%0D%0AWHERE+%7B%0D%0A++%3Fs+a+%3Ftype+.%0D%0A%7D%0D%0AGROUP+BY+%3Ftype%0D%0AORDER+BY+DESC%28%3Fnb%29&format=text%2Fhtml&should-sponge=&timeout=0&signal_void=on)):

``` sparql
SELECT DISTINCT count(?s) as ?nb ?type
WHERE {
  ?s a ?type .
}
GROUP BY ?type
ORDER BY DESC(?nb)
```

## Main entities

Pattern:

``` turtle
https://data.cimple.eu/<group>/<uuid>
# e.g. https://data.cimple.eu/claim-review/212152ebda1fabe887edb2c18046e050a44677a7d76f400e9af83d0d
```

The `<group>` is taken from this table 

| Class | Group |
| --- | --- |
| `schema:ClaimReview` | claim-review |
| `schema:Review` | review |
| `schema:SocialMediaPosting` | tweet |
| `schema:NewsArticle` | news-article |
| `schema:Organization` | organization |
| `schema:Rating` | rating |
| `schema:Claim` | claim |
| `owl:Thing` | entity |


## UUID and seed generation

The UUID is computed deterministically starting from a seed string. A real UUID taken from an example above looks like this: 212152ebda1fabe887edb2c18046e050a44677a7d76f400e9af83d0d

The seed is usually generated based on:

* Group (e.g. 'claim-review', ...)
* the id of the current object or its string value
* Hash function: SHA-2

There are some exceptions to this rule, in order to allow automatic cross-source alignment:
* For normalized ratings and birdwatch ratings, the uuid corresponds to the rating itself
* For entity, the DBpedia URL is used.

Examples of seed:
* For `claim-review`: [group]+[claim]+[normalized rating]+[review url]
* For `review` : [group]+[id]
* For `tweet`: [group] + [tweet text]
* For `news-article`: [group] + [id]
* For `organization`: [group]+[name]
* For `rating` other than normalized rating or birdwatch rating: [group] + [rating]
