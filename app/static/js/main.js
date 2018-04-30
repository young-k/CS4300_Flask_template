$(".c_title").trigger("click");

$(".more").toggle(function(){
    $(this).text("less..").siblings(".complete").show();    
}, function(){
    $(this).text("more..").siblings(".complete").hide();    
});

var $el, $ps, $up, totalHeight;
$(document).ready(function(){
	console.log("document ready");
	var list = true;
	$("#transition").click(function() {
		if (list == true){
			transition();
			list = false;
		}
		else{
			listView();
			console.log('making list');
			list = true;
		}
	});

	$(".read-more .button").click(function(){
		console.log("read more clicked");
	});

	$(".preview .button").click(function() {
		console.log("expand please");
	  totalHeight = 0

	  $el = $(this);
	  $p  = $el.parent();
	  $up = $p.parent();
	  $ps = $up.find("p:not('.read-more')");
	  
	  // measure how tall inside should be by adding together heights of all inside paragraphs (except read-more paragraph)
	  $ps.each(function() {
	    totalHeight += $(this).outerHeight() + 20;
	  });
	        
	  $up
	    .css({
	      // Set height to prevent instant jumpdown when max height is removed
	      "height": $up.height(),
	      "max-height": 9999
	    })
	    .animate({
	      "height": totalHeight
	    });
	  
	  // fade out read-more
	  $p.fadeOut();
	  
	  // prevent jump-down
	  return false;
	   
	});
});