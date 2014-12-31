var chart = nv.addGraph(function() {
    var chart = nv.models.lineChart();

    chart.xAxis
        .axisLabel("x")
        .tickFormat(d3.format(",r"));

    chart.yAxis
        .axisLabel("y")
        .tickFormat(d3.format(".02f"));

    var svg = d3.select("#vis svg");

    var tangle = new Tangle(document, {
        initialize: function() {
            this.mu = Math.PI/2;
            this.sigma = 0.5;
            this.sinYscale = 1.0;
            svg.datum(sin(this.mu, this.sigma, this.sinYscale))
              .transition().duration(500)
                .call(chart);
        },
        update:     function() {
            console.log(this.sinYscale);
            svg.datum(sin(this.mu, this.sigma, this.sinYscale))
                .call(chart);
        }
    });


    nv.utils.windowResize(function() {
        d3.select("#vis svg").call(chart);
    });

    return chart;
});


// TODO: somehow make this efficient (using crossfilter?)
function sin(mu, sigma, sinYscale) {
    var sin = [],
        x,
        xmin = -Math.PI/2.,
        xmax = Math.PI/2.,
        npts = 100,
        gaus = [];

    console.log("hi");

    var a = 1.0/(sigma * Math.sqrt(2.0 * Math.PI)),
        b = mu,
        c = sigma;

    for (var i = 0; i < npts; i++) {
        x = (xmax - xmin) * (i/npts);
        sin.push({ x: x, y: sinYscale * Math.sin(x) });
        gaus.push({ x: x, y: a * Math.exp( -( Math.pow(x-b, 2) )/( 2.0 * Math.pow(c, 2) ) ) });
    }

    return [
        {
            values: sin,
            key: "sin func",
            color: "ff7f0e"
        },
        {
            values: gaus,
            key: "gassian func",
            color: "7fff0e"
        }
    ];
}
