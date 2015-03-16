#!/bin/sh
pandoc -S --biblio publications.bib --csl ieee.csl _publications.md -t html -o tmp_body
echo "---
layout: page
title: Publications
---" > tmp_header
cat tmp_header tmp_body > index.md
rm tmp_header
rm tmp_body
