

var pc1;

d3.text("/data/sklearn_test.csv", "text/css", function(text) {

    var maxes = [];
    var org_data = d3.csv.parseRows(text);
    var data = org_data.map(function(row) {
                    var result = row.slice(1, 3)
                        .map(function(s) { return 100 * +s; });
                    return result;
               })
               .slice(0, 10);

    data = [
      [0,-0,0,0,0,3 ],
      [1,-1,1,2,1,6 ],
      [2,-2,4,4,0.5,2],
      [3,-3,9,6,0.33,4],
      [4,-4,16,8,0.25,9]
    ];

    pc1 = d3.parcoords()("#vis")
        .data(data)
        .render()
        .ticks(3)
        .createAxes();
});
