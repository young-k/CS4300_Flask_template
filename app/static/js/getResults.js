console.log(window.innerWidth);
var width = window.innerWidth, height = 2800;
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
	max = Object.keys(d).length;
	iter = 0
	for (var i = 0; i < 15; i++){
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

	views = grid(views);

	var groups = svg.selectAll(".groups")
	    .data(views)
	    .enter()
	    .append("g")
	    .attr("class", "gbar");

	groups.append("circle")
		.attr("cx", function(d, i) { return x(d['grid'][0]); })
		.attr("cy", function(d, i) { return y(d['grid'][1]); })
		.attr("r", 120)
		.attr("stroke", "#eee")
		.attr("fill", '#3D88B2')
		.attr("fill-opacity", 0.4)
		.attr("id", function(d,i) {return "node-"+i;})
		.attr("data-toggle", "modal")
        .attr("data-target", "#exampleModal")
		.on("click", function(d) {
			d3.select("#modalLabel").html(d['title']);
			d3.select(".modal-body").append("ul").attr("class", 'list-group');
			var max;
			if(d['comments'].length < 5){
				max = d['comments'].length;
			}
			else{
				max = 5;
			}
			d3.select(".modal-body").html("");
			for(var i = 0; i < max; i++){
				var item = d3.select(".modal-body").append("li").attr("class", 'list-group-item comment preview');
				item.append("p").attr("class", "comment-txt").html(d['comments'][i]['comment'])
				if(d['comments'][i]['comment'].length > 620){
					item.append("p").attr("class",'read-more').append("a").attr("class", 'button').html("Read More");
				}
				
			}
	    }); 

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