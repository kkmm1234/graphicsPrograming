using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Playershoot : MonoBehaviour
{
    public GameObject Bullet;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        Shoot();
    }

    void Shoot()
    {
        //get key down so only one cube is added
        if(Input.GetKeyDown(KeyCode.Mouse0))
        {
            //Vector3 spawnpoint = transform.position + transform.forward * 2;
            //create a bullet
            Instantiate(Bullet, transform.position, Quaternion.identity);
            Debug.Log("Shot Fired");
        }
    }
}
