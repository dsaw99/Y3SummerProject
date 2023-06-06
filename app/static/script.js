$(document).ready(function() {
    $("#setUserButton").click(function() {
        $("#loader").show();
        $("#generatedImage").hide();

        var user = $("#userInput").val();

        $.ajax({
            url: '/api/setUser',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ user: user }),
            success: function(response) {
                $("#answer").text(response);

                var newSrc = "/api/getImage?" + new Date().getTime();
                $("#generatedImage").attr("src", newSrc);

                $("#generatedImage").on("load", function() {
                    $("#loader").hide();
                    $(this).show();
                });

                $.ajax({
                    url: '/api/getScore',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ user: user }),
                    success: function(response) {
                        drawScoreChart(response.score);
                    }
                });
            }
        });
    });

    $("#chatButton").click(function() {
        var question = $("#chatInput").val();
        $.ajax({
            url: '/api/ask',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ question: question }),
            success: function(response) {
                $("#chatOutput").append('<p>' + response + '</p>');
            }
        });
    });

    $(".chat-icon").click(function() {
        var chatBox = $(".chat-box");
        if (chatBox.hasClass("visible")) {
            chatBox.removeClass("visible");
        } else {
            chatBox.addClass("visible");
        }
    });

    function drawScoreChart(score) {
        var width = 200,
            height = 200,
            radius = Math.min(width, height) / 2;

        var svg = d3.select("#scoreChart");

        var scale = d3.scaleLinear()
            .range([Math.PI / 2, 1.5 * Math.PI])
            .domain([0, 100]);

        var arc = d3.arc()
            .innerRadius(radius / 1.3)
            .outerRadius(radius)
            .startAngle(Math.PI / 2);

        svg.selectAll("*").remove();

        svg.append("path")
            .datum({ endAngle: Math.PI / 2 })
            .style("fill", "#ff9e59")
            .attr("d", arc)
            .attr("transform", "translate(" + width / 2 + "," + height + ")")
            .transition()
            .duration(2000)
            .call(arcTween, scale(score));

        svg.append("text")
            .attr("x", width / 2)
            .attr("y", height / 1.5)
            .attr("text-anchor", "middle")
            .style("font-size", "24px")
            .text(score);

        function arcTween(transition, newAngle) {
            transition.attrTween("d", function(d) {
                var interpolate = d3.interpolate(d.endAngle, newAngle);
                return function(t) {
                    d.endAngle = interpolate(t);
                    return arc(d);
                };
            });
        }
    }
});