var MongoClient = require('mongodb').MongoClient;
var url = "mongodb://meta:password@localhost:27017/serverless-metrics";
// admin:adminpassword
// meta:password

const args = process.argv.slice(2);
console.log('Arguments: ', args);
if (args.length != 1) {
    return;
}

const fastcsv = require("fast-csv");
const fs = require("fs");

const tableName = args[0];
const fileName = tableName.concat(".csv");
const dbName = "serverless-metrics";

MongoClient.connect(
  url,
  { useNewUrlParser: true, useUnifiedTopology: true },
  (err, client) => {
    if (err) throw err;

    client
      .db(dbName)
      .collection(tableName)
      .find({})
      .toArray((err, data) => {
        if (err) throw err;
	if (data.length === 0) {
	  console.log("No data!")
	} else {
	  const ws = fs.createWriteStream(fileName);
          // console.log(data);
          fastcsv
            .write(data, { headers: true })
            .on("finish", function() {
              console.log("Exported successfully!");
            })
            .pipe(ws);
	}
	client.close();
      });
  }
);
