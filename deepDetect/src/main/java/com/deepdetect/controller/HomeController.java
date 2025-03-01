package com.deepdetect.controller;


import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;

@Controller
public class HomeController {

    @GetMapping("/")
    public String home() {
        return "index";
    }
    @GetMapping("/upload")
    public String upload() {
        return "upload";
    }
    @GetMapping("/about")
    public String about() {
        return "about";
    }
    @GetMapping("/insight")
    public String insight() {
        return "insight";
    }


}
