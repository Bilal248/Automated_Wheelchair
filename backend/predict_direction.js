const { exec } = require("child_process");

function predictDirection(sensorData, callback) {
  const args = [
    sensorData.timeOfFlight,
    sensorData.hc,
    sensorData.latitude,
    sensorData.longitude,
    sensorData.speed,
    sensorData.bearing
  ].join(" ");

  exec(`python3 ai_model/predict.py ${args}`, (error, stdout, stderr) => {
    if (error) return callback(error, null);
    callback(null, stdout.trim());
  });
}

module.exports = predictDirection;