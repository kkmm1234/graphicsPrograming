using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SphereMovement : MonoBehaviour
{
    public float speed = 2.0f;
    private Rigidbody rb;
    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void Update()
    {
        // Move the sphere forward
        if (Input.GetKey(KeyCode.UpArrow))
        {
            rb.AddForce(Vector3.forward * speed);
            Debug.Log("Moving forward");
        }
        // Move the sphere backward
        if(Input.GetKey(KeyCode.DownArrow))
        {
            rb.AddForce(Vector3.back * speed);
            Debug.Log("Moving backward");
        }
        // Move the sphere left
        if(Input.GetKey(KeyCode.LeftArrow))
        {
            rb.AddForce(Vector3.left * speed);
            Debug.Log("Moving left");
        }
        // Move the sphere right
        if(Input.GetKey(KeyCode.RightArrow))
        {
            rb.AddForce(Vector3.right * speed);
            Debug.Log("Moving right");
        }
    }
}
