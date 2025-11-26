//SPDX-License-Identifier: MIT

pragma solidity ^0.8.0;

contract Reputation {
    address public owner;
    mapping(address => uint) public trustScores;

    constructor() {
        owner = msg.sender;
    }

    function setTrust(address vehicle, uint trust) public {
        require(msg.sender == owner, "Only owner can set trust");
        trustScores[vehicle] = trust;
    }

    function getTrust(address vehicle) public view returns (uint) {
        return trustScores[vehicle];
    }
}
