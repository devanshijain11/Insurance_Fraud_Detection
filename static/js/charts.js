function loadCharts(fraudData, vehicleData){

new Chart(document.getElementById("fraudChart"), {
    type: "pie",
    data: {
        labels: Object.keys(fraudData),
        datasets: [{
            data: Object.values(fraudData)
        }]
    }
});

new Chart(document.getElementById("vehicleChart"), {
    type: "bar",
    data: {
        labels: Object.keys(vehicleData),
        datasets: [{
            label: "Fraud Cases",
            data: Object.values(vehicleData)
        }]
    }
});

}