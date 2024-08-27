using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    public float speedH = 2.0f;
    public float speedV = 2.0f;

    private float yaw = 0.0f;
    private float pitch = 0.0f;
    public GameObject Player;
    private Vector3 offset;
    // Start is called before the first frame update
    void Start()
    {
    //find position of camera relative to player
       offset = transform.position - Player.transform.position; 
    }

    // LateUpdate is called after Update each frame
    void LateUpdate()
    {
        //rotate camera based on mouse movement
        yaw += speedH * Input.GetAxis("Mouse X");
        pitch -= speedV * Input.GetAxis("Mouse Y");

        transform.eulerAngles = new Vector3(pitch, yaw, 0.0f);
        
        //update camera position to follow player
        transform.position = Player.transform.position + offset;
    }
}
