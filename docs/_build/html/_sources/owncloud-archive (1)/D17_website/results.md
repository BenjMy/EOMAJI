---
layout: default
title: "Results"
rank: 5
---

# Results
Here we post the most recent activities and results, ordered in chronological order from newest to oldest.

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      {{ post.excerpt }}
    </li>
  {% endfor %}
</ul>
