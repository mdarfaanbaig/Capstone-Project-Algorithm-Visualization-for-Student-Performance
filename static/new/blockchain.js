var contract;
ethereum.enable();
var address="0x026367c7397Cb0BA22edBd613af8ecd8859D324a";
var gasPriceval="3";
var gasval="300";
$(document).ready(function(){
	
	web3=new Web3(web3.currentProvider);
	//var address="0x64ADd870Fb9d6DbdA79504a7458CaDDF6CfE74da";
	var abi=[
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "filename",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "uploader",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "privatekey",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "encryptedkey",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "publickey",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "ipfsfolderhash",
				"type": "string"
			}
		],
		"name": "addFiledetails",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "patientKey",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "phi",
				"type": "string"
			}
		],
		"name": "addHealthRecord",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "providerKey",
				"type": "string"
			}
		],
		"name": "authorizeProvider",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": false,
				"internalType": "string",
				"name": "access",
				"type": "string"
			}
		],
		"name": "checkAccess",
		"type": "event"
	},
	{
		"inputs": [],
		"name": "checkNextRegistry",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": false,
				"internalType": "bool",
				"name": "returnValue",
				"type": "bool"
			}
		],
		"name": "checkRegistry",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "uploader",
				"type": "string"
			}
		],
		"name": "getfiledetails",
		"outputs": [
			{
				"components": [
					{
						"internalType": "address",
						"name": "key",
						"type": "address"
					},
					{
						"internalType": "string",
						"name": "filename",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "uploader",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "privatekey",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "encryptedkey",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "publickey",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "ipfsfolderhash",
						"type": "string"
					}
				],
				"internalType": "struct MedicalBlock.filedetails[]",
				"name": "",
				"type": "tuple[]"
			}
		],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "filename",
				"type": "string"
			}
		],
		"name": "getfiledetailsfromfilename",
		"outputs": [
			{
				"components": [
					{
						"internalType": "address",
						"name": "key",
						"type": "address"
					},
					{
						"internalType": "string",
						"name": "filename",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "uploader",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "privatekey",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "encryptedkey",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "publickey",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "ipfsfolderhash",
						"type": "string"
					}
				],
				"internalType": "struct MedicalBlock.filedetails[]",
				"name": "",
				"type": "tuple[]"
			}
		],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": false,
				"internalType": "string",
				"name": "data",
				"type": "string"
			}
		],
		"name": "getRecord",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "receiver",
				"type": "string"
			}
		],
		"name": "getShareFile",
		"outputs": [
			{
				"components": [
					{
						"internalType": "address",
						"name": "key",
						"type": "address"
					},
					{
						"internalType": "string",
						"name": "filname",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "uploader",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "receiver",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "ipfsfolderhash",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "encry",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "privatekey",
						"type": "string"
					}
				],
				"internalType": "struct MedicalBlock.Sharefile[]",
				"name": "",
				"type": "tuple[]"
			}
		],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "patientKey",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "recordIndex",
				"type": "uint256"
			}
		],
		"name": "getThisHealthRecord",
		"outputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			}
		],
		"name": "getUserdata",
		"outputs": [
			{
				"components": [
					{
						"internalType": "address",
						"name": "key",
						"type": "address"
					},
					{
						"internalType": "string",
						"name": "name",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "dob",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "mobile",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "email",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "password",
						"type": "string"
					}
				],
				"internalType": "struct MedicalBlock.userinfo[]",
				"name": "",
				"type": "tuple[]"
			}
		],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "getUserType",
		"outputs": [
			{
				"internalType": "int256",
				"name": "",
				"type": "int256"
			}
		],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "requester",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "patientKey",
				"type": "string"
			}
		],
		"name": "isAuthorized",
		"outputs": [
			{
				"internalType": "bool",
				"name": "",
				"type": "bool"
			}
		],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "name",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "password",
				"type": "string"
			}
		],
		"name": "Login",
		"outputs": [
			{
				"components": [
					{
						"internalType": "address",
						"name": "key",
						"type": "address"
					},
					{
						"internalType": "string",
						"name": "name",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "dob",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "mobile",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "email",
						"type": "string"
					},
					{
						"internalType": "string",
						"name": "password",
						"type": "string"
					}
				],
				"internalType": "struct MedicalBlock.userinfo[]",
				"name": "",
				"type": "tuple[]"
			}
		],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "addrString",
				"type": "string"
			}
		],
		"name": "registerPatient",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "addrString",
				"type": "string"
			}
		],
		"name": "registerProvider",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "filname",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "uploader",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "receiver",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "ipfsfolderhash",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "encry",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "privatekey",
				"type": "string"
			}
		],
		"name": "ShareFile",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "user",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "name",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "dob",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "mobile",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "email",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "password",
				"type": "string"
			}
		],
		"name": "userRegister",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"anonymous": false,
		"inputs": [
			{
				"indexed": false,
				"internalType": "int256",
				"name": "uType",
				"type": "int256"
			}
		],
		"name": "userType",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "allfiledetailss",
		"outputs": [
			{
				"internalType": "address",
				"name": "key",
				"type": "address"
			},
			{
				"internalType": "string",
				"name": "filename",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "uploader",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "privatekey",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "encryptedkey",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "publickey",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "ipfsfolderhash",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "allSharefiles",
		"outputs": [
			{
				"internalType": "address",
				"name": "key",
				"type": "address"
			},
			{
				"internalType": "string",
				"name": "filname",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "uploader",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "receiver",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "ipfsfolderhash",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "encry",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "privatekey",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "uint256",
				"name": "",
				"type": "uint256"
			}
		],
		"name": "allusers",
		"outputs": [
			{
				"internalType": "address",
				"name": "key",
				"type": "address"
			},
			{
				"internalType": "string",
				"name": "name",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "dob",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "mobile",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "email",
				"type": "string"
			},
			{
				"internalType": "string",
				"name": "password",
				"type": "string"
			}
		],
		"stateMutability": "view",
		"type": "function"
	}
];
	contract=new web3.eth.Contract(abi,address);
})
