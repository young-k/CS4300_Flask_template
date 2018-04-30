max = Object.keys(views).length;
rows = max/3 + 1;
var width = window.innerWidth, height = 270*(max/3 + 1);
var svg = d3.select("body").append("svg")
  .attr("width", width)
  .attr("height", height);

var x = d3.scaleLinear()
  .range([300, width-300])
  .domain([0, 2]);

var y = d3.scaleLinear()
  .range([150, 2700])
  .domain([0, 10]);

function grid(d){
  iter = 0
  for (var i = 0; i < rows; i++){
    for (var j = 0; j < 3; j++){
      if (iter >= max){
        return d;
      }
      d[iter]['grid'] = [j, i];
      iter++;
    }
  }
  return d;
}

function openMod(d){
  console.log('hi')
  console.log(d);
  d3.select("#modalLabel").html(d['title']);
  d3.select("#postURL").attr("href", d['url']);
  d3.select("#postContent").html(d['body']);
  d3.select(".modal-body").append("ul").attr("class", 'list-group');
  var max;
  if(d['top_comments'].length < 5){
    max = d['top_comments'].length;
  }
  else{
    max = 5;
  }
  d3.select(".modal-body").html("");
  for(var i = 0; i < max; i++){
    var item = d3.select(".modal-body").append("li").attr("class", 'list-group-item comment preview');
    item.append("p").attr("class", "comment-txt").html(d['top_comments'][i]['html_body'])
    if(d['top_comments'][i]['body'].length > 620){
      var p = item.append("p").attr("class",'read-more');
      p.append("a").attr("class", 'button')
        .html("[+]")
        .on("mouseover", function(){ d3.select(this).style("cursor", "pointer"); })
        .on("click", function(d) {
          //functionality of expanding comments
          totalHeight = 0
          $el = $(this);
          $p  = $el.parent();
          $up = $p.parent();
          $ps = $up.find("p:not('.read-more')");

          // measure how tall inside should be by adding together heights of all inside paragraphs (except read-more paragraph)
          $ps.each(function() {
            totalHeight += $(this).outerHeight() + 20;
          });

          $up.css({
            // Set height to prevent instant jumpdown when max height is removed
            "max-height": 9999
          })
          // fade out read-more
          $p.fadeOut();
          // prevent jump-down
          return false;	
        });
    }
  }
}

views = grid(views);

var groups = svg.selectAll(".groups")
  .data(views)
  .enter()
  .append("g")
  .attr("class", "gbar")
  .attr("data-toggle", "modal")
  .attr("data-target", "#exampleModal")
  .on("mouseover", function () {
    console.log('mouseover')
    var t = d3.select(this);
    t.select("circle").attr("stroke", "#656565");
    t.style("cursor", "pointer"); 
  })
  .on("mouseout", function () {
    d3.select(this).select("circle")
      .attr("stroke",  "#ddd");
  })
  .on("click", function(d) {
    console.log('clicked')
    openMod(d);		
  }); 

groups.append("circle")
  .attr("cx", function(d, i) { return x(d['grid'][0]); })
  .attr("cy", function(d, i) { return y(d['grid'][1]); })
  .attr("r", 120)
  .attr("fill", '#3D88B2')
  .attr("fill-opacity", 0.4)
  .attr("stroke", "#ddd")
  .attr("stroke-width", 4)
  .attr("id", function(d,i) {return "node-"+i;}); 

groups.append("text")
  .attr("x", function(d, i) { return x(d['grid'][0])-75; })
  .attr("y", function(d, i) { return y(d['grid'][1])-75; })
  .attr("width", 150)
  .attr("height", 150)
  .text(function(d){return d['title']})
  .attr("id", function(d,i) {return "text-"+i;}); 

for(var i = 0; i <id; i++){
  d3plus.textwrap()
    .container('text#text-'+i) 
    .valign("middle")
    .shape("circle")
    .draw();
}
// });
