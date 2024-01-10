/*
SQLyog Community Edition- MySQL GUI v7.01 
MySQL - 5.0.27-community-nt : Database - algorithmvisualizer
*********************************************************************
*/


CREATE DATABASE /*!32312 IF NOT EXISTS*/`algorithmvisualizer` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `algorithmvisualizer`;

/*Table structure for table `register` */

DROP TABLE IF EXISTS `register`;

CREATE TABLE `register` (
  `id` int(255) NOT NULL auto_increment,
  `username` varchar(255) default NULL,
  `email` varchar(255) default NULL,
  `password` varchar(255) default NULL,
  `mobileno` varchar(255) default NULL,
  `address` varchar(255) default NULL,
  PRIMARY KEY  (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

/*Data for the table `register` */

insert  into `register`(`id`,`username`,`email`,`password`,`mobileno`,`address`) values (1,'a','yashsalvi1999@gmail.com','a','9930090883','vedika soc sagar nagar vikhroli park site'),(2,'a','a@gmail.com','a','5845458545','a');


