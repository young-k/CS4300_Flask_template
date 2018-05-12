max = Object.keys(views).length;
rows = max/3 + 1;
var openedPost, groups;
var width = window.innerWidth, height = 270*(max/3 + 1);

var div = d3.select("body").append("div")	
	    .attr("class", "tooltip")				
	    .style("opacity", 0);

var svg = d3.select("body").append("svg")
	.attr("width", width)
	.attr("height", height);

var x = d3.scaleLinear()
	.range([300, width-300])
	.domain([0, 2]);

var y = d3.scaleLinear()
	.range([150, 2700])
	.domain([0, 10]);

var defs = svg.append("defs");

//Not used anymore
defs.append("radialGradient")
	.attr("id", "result-gradient")
	.attr("cx", "50%")	//not really needed, since 50% is the default
	.attr("cy", "50%")	//not really needed, since 50% is the default
	.attr("r", "50%")	//not really needed, since 50% is the default
	.selectAll("stop")
	.data([
			{offset: "0%", color: "#1aaaf0"},
			{offset: "50%", color: "#4bbdf4"},
			{offset: "90%", color: "#9cdbf9"}, //#4bbdf4
			{offset: "100%", color: "#ffffff"}
		])
	.enter().append("stop")
	.attr("offset", function(d) { return d.offset; })
	.attr("stop-color", function(d) { return d.color; });

d3.selection.prototype.moveToFront = function() {  
      return this.each(function(){
        this.parentNode.appendChild(this);
      });
    };

d3.selection.prototype.moveToBack = function() {  
    return this.each(function() { 
        var firstChild = this.parentNode.firstChild; 
        if (firstChild) { 
            this.parentNode.insertBefore(this, firstChild); 
        } 
    });
};

//Grid gives each result entry a coordinate that will be used when finding the correct placement in the SVG
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

//transitioning from list view to cluster view
function transition(){
	var coords = []
	var max = views.length;
	svg.attr("height", 700);
	for(var i = 0; i < max; i++){
		coords = coords.concat(views[i]['coordinate']);
	}
	coords = coords.concat(opinion_coordinates);
	
	var coordMax = Math.max(Math.abs(d3.max(coords)), Math.abs(d3.min(coords)));
	var clusterPadding = Math.random()+0.5; //unecessary now
	var newX = d3.scaleLinear()
		.range([150, width-150])
		.domain([-1*coordMax-clusterPadding, coordMax+clusterPadding]);

	var newY = d3.scaleLinear()
		.range([20, 400])
		.domain([-1*coordMax-clusterPadding, coordMax+clusterPadding]);

	var xAxis = d3.svg.axis().scale(newX);
	var yAxis = d3.svg.axis().scale(newY);

	console.log("transitioning to cluster view");

	var updateCircles = groups.selectAll("circle");
	var updateTexts = groups.selectAll("text");

	updateTexts.transition().duration(200).remove();

	updateCircles
		.transition().duration(1200)
		.attr("cx", function(d){return newX(d['coordinate'][0]);})
		.attr("cy", function(d){return newY(d['coordinate'][1]);})
		.attr("stroke-width", 2)
		.attr("r", 30);

	updateCircles
		.on("mouseover", function(d){
			var c = d3.select(this);
			c.attr("stroke",  "#ddd")
			.attr("stroke-width", 4)
			.moveToFront();

			div.style("opacity", 0.9);

			div.html(d['title'])	
            .style("left", (d3.event.pageX + 30) + "px")		
            .style("top", (d3.event.pageY - 28) + "px");	
		})
		.on("mouseout", function(d){
			var c = d3.select(this);
			c.attr("stroke",  "none")
			.attr("fill-opacity", 0.5)
			.attr("stroke-width", 2);

			div.style("opacity", 0);	
		});
	//If there is an opinion, then append the opinion circle
	if(s != "N/A"){
		var op = svg.append("circle").attr("fill-opacity", 0.0);
		op.transition().duration(1200)
			.attr("cx", newX(opinion_coordinates[0]))
			.attr("cy", newY(opinion_coordinates[1]))
			.attr("id", "opinion-point")
			.attr("r", 30)
			.style("fill", "grey")
			.attr("fill-opacity", 0.5);

		op.on("mouseover", function(){
			var c = d3.select(this);

			c.attr("stroke", "white")
			.attr("stroke-width", 4);

			div.style("opacity", 0.9);
			div.html(s)	
	        .style("left", (d3.event.pageX + 30) + "px")		
	        .style("top", (d3.event.pageY - 28) + "px");
		})
		.on("mouseout", function(d){
				var c = d3.select(this);
				c.attr("stroke",  "none")
				.attr("fill-opacity", 0.5)
				.attr("stroke-width", 2);

				div.style("opacity", 0);	
		});;	
		op.moveToBack();
	}
}

//Transitioning from list biew to cluster view 
function listView(){
	svg.attr("height", 2700);
	var updateCircles = groups.selectAll("circle");
	// var updateTexts = groups.selectAll("text");
	updateCircles
		.transition().duration(1200)
		.attr("cx", function(d, i) { return x(d['grid'][0]); })
		.attr("cy", function(d, i) { return y(d['grid'][1]); })
		.attr("r", 120)
		.attr("stroke-width", 4);

	updateCircles.on('mouseover', null);

	groups.append("text").transition()
	.attr("x", function(d, i) { return x(d['grid'][0])-75; })
	.attr("y", function(d, i) { return y(d['grid'][1])-75; })
	.attr("width", 150)
	.attr("height", 150)
	.text(function(d){return d['title'];})
	.attr("id", function(d,i) {return "text-"+i;})
	.attr("opacity", 0.0)
	.style("font-size", "16px");

	d3.select("#opinion-point").remove();

	setTimeout(function(){
		for(var i = 0; i <id; i++){
			d3plus.textwrap()
			  .container('text#text-'+i) 
			  .valign("middle")
			  .shape("circle")
			  .draw();
			d3.select('#text-'+i).transition().duration(1000).attr("opacity", 1.0);
		}}, 
	1200)

}

function openMod(d){
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

function createResults(){
	views = grid(views);

	groups = svg.selectAll(".groups")
	    .data(views)
	    .enter()
	    .append("g")
	    .attr("class", "gbar")
	    .attr("data-toggle", "modal")
	    .attr("data-target", "#exampleModal")
	    .on("mouseover", function () {
	        // console.log('mouseover')
	    	var t = d3.select(this);
	    	t.select("circle").attr("stroke", "#656565");
	    	t.style("cursor", "pointer"); 
	    })
	    .on("mouseout", function () {
	    	d3.select(this).select("circle")
	    	.attr("stroke",  "#ddd");
	    })
	    .on("click", function(d) {
			openMod(d);	
			openedPost = d;
		}); 

	groups.append("circle")
		.attr("cx", function(d, i) { return x(d['grid'][0]); })
		.attr("cy", function(d, i) { return y(d['grid'][1]); })
		.attr("r", 120)
		.attr("fill", '#3D88B2')
		.attr("fill-opacity", 0.5)
		.attr("stroke", "#ddd")
		.attr("stroke-width", 4)
		.attr("id", function(d,i) {return "node-"+i;});

	groups.append("text")
		.attr("x", function(d, i) { return x(d['grid'][0])-75; })
		.attr("y", function(d, i) { return y(d['grid'][1])-75; })
		.attr("width", 150)
		.attr("height", 150)
		.style("font-size", "16px")
		.text(function(d){return d['title']})
		.attr("id", function(d,i) {return "text-"+i;}); 

	for(var i = 0; i <id; i++){
		d3plus.textwrap()
		  .container('text#text-'+i) 
		  .valign("middle")
		  .shape("circle")
		  .draw();
	}
}

