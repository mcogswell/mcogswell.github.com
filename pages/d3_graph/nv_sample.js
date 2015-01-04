var startNodes = [{}],
    startLinks = [];






var width = 960,
    height = 500;

var fill = d3.scale.category20();

var force = d3.layout.force()
    .size([width, height])
    .nodes(startNodes) // initialize with a single node
    .links(startLinks)
    .linkDistance(30)
    .charge(-60)
    .on("tick", tick);

var tree = d3.layout.tree()
    .size([width, height])
    .nodes(startNodes)
    .links(startLinks)
    .on("tick", tick);

var svg = d3.select("#vis").append("svg")
    .attr("width", width)
    .attr("height", height)
    .on("mousemove", mousemove)
    .on("mousedown", mousedown);

svg.append("rect")
    .attr("width", width)
    .attr("height", height);

var nodes = force.nodes(),
    links = force.links(),
    node = svg.selectAll(".node"),
    link = svg.selectAll(".link");

var hovered = null;

var cursor = svg.append("circle")
    .attr("r", 30)
    .attr("transform", "translate(-100,-100)")
    .attr("class", "cursor");

restart();

function mousedown() {
    if (hovered === null) {
        createNode.apply(this);
    }
    else {
        d3.select(hovered)
            .each(function(d) { d.cls = (d.cls + 1) % 2; })
            .classed("link1", function(d) { return d.cls == 0; })
            .classed("link2", function(d) { return d.cls == 1; });
    }
}

function mousemove() {
  cursor.attr("transform", "translate(" + d3.mouse(this) + ")");
}

// mouseenter and mouseleave are equivalent to css :hover...
// http://www.quirksmode.org/dom/events/index.html
function mouseover() {
    hovered = this;
}

function mouseout() {
    hovered = null;
}

function createNode() {
  var point = d3.mouse(this),
      node = {x: point[0], y: point[1]},
      n = nodes.push(node);

  // add links to any nearby nodes
  nodes.forEach(function(target) {
    var x = target.x - node.x,
        y = target.y - node.y;
    if (Math.sqrt(x * x + y * y) < 30) {
      links.push({source: node, target: target, cls: 0});
    }
  });

  restart();
}

function tick() {
  link.attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; });

  node.attr("cx", function(d) { return d.x; })
      .attr("cy", function(d) { return d.y; });
}

function restart() {
  link = link.data(links);

  link.enter().insert("line", ".node")
      // TODO: how to CSS?
      .attr("class", "link link1")
      .on("mouseover", mouseover)
      .on("mouseout", mouseout);

  node = node.data(nodes);

  node.enter().insert("circle", ".cursor")
      .attr("class", "node")
      .attr("r", 6)
      .call(force.drag);

  force.start();
}
