
var dataList = document.getElementById('airport-datalist');
var input1 = document.getElementById('origin');
var input2 = document.getElementById('dest');

// Create a new XMLHttpRequest.
var request = new XMLHttpRequest();

// Handle state changes for the request.
request.onreadystatechange = function(response) {
	if (request.readyState === 4) {
	    if (request.status === 200) {
	      // Parse the JSON
	      var jsonOptions = JSON.parse(request.responseText);

	      // Loop over the JSON array.
	      jsonOptions.forEach(function(item) {
	        // Create a new <option> element.
	        var option = document.createElement('option');
	        // Set the value using the item in the JSON array.
	        option.innerText = item.DESCRIPTION;
	        option.value = item.AIRPORT_CODE;
	        // Add the <option> element to the <datalist>.
	        dataList.appendChild(option);
	      });

	      // Update the placeholder text.
	      input1.placeholder = "e.g. city name";
	      input2.placeholder = "e.g. city name";
	    } else {
	      // An error occured :(
	      input1.placeholder = "Couldn't load datalist options :(";
	      input2.placeholder = "Couldn't load datalist options :(";
	    }
	}
};

// Update the placeholder text.
input1.placeholder = "Loading airports...";
input2.placeholder = "Loading airports...";

// Set up and make the request.
request.open('GET', airportListpath, true);
request.send();